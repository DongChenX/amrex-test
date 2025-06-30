#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>
#include <cmath>
#include "ParticleTest.H"

#include<iostream>

using namespace amrex;

# define PI 3.1415926535

void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal){
    //pvf是cell-centre的网格
    //phi_nodal是nodal-bases的网格
    std::cout<<"In the nodal_phi_to_pvf"<<std::endl;

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (Gpu::notInLaunchRegion())
    #endif

    for (MFIter mfi(pvf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& pvffab = pvf.array(mfi);
        auto const& pnffab = phi_nodal.array(mfi);
        amrex::ParallelFor(bx, [pvffab, pnffab]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            Real num = 0.0;
            for(int ii = i; ii<=i+1; ii++)
            {
                for(int jj = j; jj<=j+1; jj++)
                {
                    for(int kk = k;kk<=k+1; kk++)
                    {
                        num += -1.0*pnffab(ii,jj,kk)*nodal_phi_to_heavi(pnffab(ii,jj,kk));
                    }
                }
            }

            Real re = 0.0;
            for(int ii = i; ii<=i+1; ii++)
            {
                for(int jj = j; jj<=j+1; jj++)
                {
                    for(int kk = k;kk<=k+1; kk++)
                    {
                        //re += amrex::Math
                        re += amrex::Math::abs(pnffab(ii,jj,kk));
                    }
                }
            }

            pvffab(i,j,k) = num/(re + 1e-12);
        }
        );

    }
    
}



//deltafunction
AMREX_FORCE_INLINE
void deltaFunction(Real xf, Real xp, Real h, Real& value, DELTA_FUNCTION_TYPE type)
{
    Real rr = amrex::Math::abs(( xf - xp ) / h);
    switch (type) {
    case DELTA_FUNCTION_TYPE::FOUR_POINT_IB:
        if(rr >= 0 && rr < 0.5 ){
            value = 1.0 / 8.0 * ( 3.0 - 2.0 * rr + std::sqrt( 1.0 + 4.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else if (rr >= 1 && rr < 2) {
            value = 1.0 / 8.0 * ( 5.0 - 2.0 * rr - std::sqrt( -7.0 + 12.0 * rr - 4 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    case DELTA_FUNCTION_TYPE::THREE_POINT_IB:
        if(rr >= 0 && rr < 1){
            value = 1.0 / 6.0 * ( 5.0 - 3.0 * rr + std::sqrt( - 3.0 * ( 1 - Math::powi<2>(rr)) + 1.0 )) / h;
        }else if (rr >= 1 && rr < 2) {
            value = 1.0 / 3.0 * ( 1.0 + std::sqrt( 1.0 - 3 * Math::powi<2>(rr))) / h;
        }else {
            value = 0;
        }
        break;
    default:
        break;
    }
}



//其他有用的一些函数
[[nodiscard]] AMREX_FORCE_INLINE
//计算一个球的转动惯量
Real cal_momentum(Real rho, Real radious)
{
    return  (8.0/15.0)*PI*rho*radious*radious*radious*radious*radious;
}


//将力插值到临近的欧拉点上
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ForceSpreading_cic (P const& p,
                  ParticleReal fxP,
                  ParticleReal fyP,
                  ParticleReal fzP,
                  Array4<Real> const& E,
                  int EularFIndex,
                  GpuArray<Real,AMREX_SPACEDIM> const& plo,
                  GpuArray<Real,AMREX_SPACEDIM> const& dx,
                  DELTA_FUNCTION_TYPE type)
{
    const Real d = dx[0]*dx[1]*dx[2]; //计算欧拉网格的体积

    //计算拉格朗日点所在的网格
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];

    //向下取整，获取
    int index_i = static_cast<int>(std::floor(lx));
    int index_j = static_cast<int>(std::floor(ly));
    int index_k = static_cast<int>(std::floor(lz));

    for(int ii = index_i-2;ii<=index_i+2;ii++)
    {
        for(int jj = index_j-2;jj<=index_j+2;jj++)
        {
            for(int kk = index_k-2;kk<=index_k+2;kk++)
            {
                //计算欧拉网格的中心坐标
                
            }
        }
    }

}