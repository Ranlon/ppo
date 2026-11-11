"""Inference-only RMSNorm kernels with explicit BF16 boundary; never patch training."""
import contextlib,types
import torch
import triton
import triton.language as tl

def native(x,w,eps):
    value=x.float();variance=value.pow(2).mean(-1,keepdim=True)
    return w*(value*torch.rsqrt(variance+eps)).to(x.dtype)

@triton.jit
def square_kernel(X,S,TOTAL,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    v=tl.load(X+i,i<TOTAL,other=0).to(tl.float32)
    tl.store(S+i,v*v,i<TOTAL)

@triton.jit
def post_kernel(X,W,V,Y,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,BLOCK)
    x=tl.load(X+row*N+i,i<N,other=0).to(tl.float32)
    variance=tl.load(V+row).to(tl.float32)
    inv=tl.rsqrt(variance+EPS)
    rounded=(x*inv).to(X.dtype.element_ty).to(tl.float32)
    w=tl.load(W+i,i<N,other=0).to(tl.float32)
    tl.store(Y+row*N+i,rounded*w,i<N)

@torch.library.custom_op('m3rms::staged',mutates_args=())
def staged(x:torch.Tensor,w:torch.Tensor,eps:float)->torch.Tensor:
    if x.device.type!='cuda':return native(x,w,eps)
    assert x.is_contiguous() and w.is_contiguous()
    n=x.shape[-1];rows=x.numel()//n
    squared=torch.empty_like(x,dtype=torch.float32,memory_format=torch.contiguous_format)
    square_kernel[(triton.cdiv(x.numel(),256),)](x,squared,x.numel(),BLOCK=256,enable_fp_fusion=False)
    variance=squared.mean(-1,keepdim=True) # Exactly the original ATen reduction tree.
    out=torch.empty_like(x,dtype=torch.promote_types(x.dtype,w.dtype),memory_format=torch.contiguous_format)
    post_kernel[(rows,)](x,w,variance,out,n,eps,triton.next_power_of_2(n),num_warps=4,enable_fp_fusion=False)
    return out

def fake(x,w,eps):return torch.empty_like(x,dtype=torch.promote_types(x.dtype,w.dtype),memory_format=torch.contiguous_format)
staged.register_fake(fake)

class NormPatch:
    def __init__(self,model,mode):
        self.entries=[];assert mode=='staged';op=staged
        for m in model.modules():
            if type(m).__name__!='Qwen3RMSNorm':continue
            def factory(operation):
                def forward(this,x):return operation(x,this.weight,float(this.variance_epsilon))
                return forward
            self.entries.append((m,types.MethodType(factory(op),m),'forward' in m.__dict__,m.__dict__.get('forward')))
        assert len(self.entries)==145
    @contextlib.contextmanager
    def active(self):
        try:
            for m,fn,_,_ in self.entries:m.forward=fn
            yield
        finally:
            for m,_,had,old in self.entries:
                if had:m.forward=old
                else:del m.forward
