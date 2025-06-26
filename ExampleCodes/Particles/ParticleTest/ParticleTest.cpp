#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>

#include "ParticleTest.H"

#include<iostream>

using namespace amrex;


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