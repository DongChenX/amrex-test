#ifndef PARTICLE_TEST_H
#define PARTICLE_TEST_H

#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>

using namespace amrex;

AMREX_INLINE AMREX_GPU_DEVICE //启动GPU并行计算

//计算颗粒体积分数辅助函数
Real nodal_phi_to_heavi(Real phi)
{
    if (phi <= 0)
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}

//这一个用来计算相分数的函数
void nodal_phi_to_pvf(MultiFab& pvf, const MultiFab& phi_nodal);

enum DELTA_FUNCTION_TYPE{
    FOUR_POINT_IB,
    THREE_POINT_IB
};

//定义一个颗粒质心数据的结构提
struct Kernel{
    RealVect velocity;
    RealVect location;
    RealVect omega; // 该物理量含义存疑
    RealVect varphi; //该物理量作用存疑
    Real rho;
    Real radious;
    Real ml; //暂时存疑
    Real dv; //颗粒体积
};


#endif // 