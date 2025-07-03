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
                  int EulerFIndex,
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

    //index_i index_j index_k都是拉式点所在的网格
    for(int ii = index_i-2;ii<=index_i+2;ii++)
    {
        for(int jj = index_j-2;jj<=index_j+2;jj++)
        {
            for(int kk = index_k-2;kk<=index_k+2;kk++)
            {
                //计算欧拉网格的中心坐标
                Real tu,tv,tw;
                Real xf_temp = ii*dx[0] + 0.5*dx[0];
                Real yf_temp = jj*dx[1] + 0.5*dx[1];
                Real zf_temp = kk*dx[2] + 0.5*dx[2];
                deltaFunction(p.pos(0), xf_temp, dx[0], &tu, type);
                deltaFunction(p.pos(1), yf_temp, dx[1], &tv, type);
                deltaFunction(p.pos(2), zf_temp, dx[2], &tw, type);
                Real dU = tu*tv*tw*d;
                Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex), fxP*dU);
                Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex+1), fyP*dU);
                Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex+2), fzP*dU);
            }
        }
    }

}


//这里是将欧式点的速度插值到拉式点上面
emplate <typename P = Particle<numAttri>>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void VelocityInterpolation_cir(P const& p, Real& Up, Real& Vp, Real& Wp,
                     Array4<Real const> const& E, int EulerVIndex,
                     GpuArray<Real, AMREX_SPACEDIM> const& plo,
                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                     DELTA_FUNCTION_TYPE type)
{
    const Real d = dx[0]*dx[1]*dx[2]; //计算欧拉网格的体积
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];

    //向下取整，获取
    int index_i = static_cast<int>(std::floor(lx));
    int index_j = static_cast<int>(std::floor(ly));
    int index_k = static_cast<int>(std::floor(lz));
    //给颗粒的速度赋予0值，避免干扰
    Up = 0;
    Vp = 0;
    Wp = 0;
    //index_i index_j index_k都是拉式点所在的网格
    for(int ii = index_i-2;ii<=index_i+2;ii++)
    {
        for(int jj = index_j-2;jj<=index_j+2;jj++)
        {
            for(int kk = index_k-2;kk<=index_k+2;kk++)
            {
                //计算欧拉网格的中心坐标
                Real tu,tv,tw;
                Real xf_temp = ii*dx[0] + 0.5*dx[0];
                Real yf_temp = jj*dx[1] + 0.5*dx[1];
                Real zf_temp = kk*dx[2] + 0.5*dx[2];
                deltaFunction(p.pos(0), xf_temp, dx[0], &tu, type);
                deltaFunction(p.pos(1), yf_temp, dx[1], &tv, type);
                deltaFunction(p.pos(2), zf_temp, dx[2], &tw, type);
                Real dU = tu*tv*tw*d;
                //Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerVIndex), Up*dU);
                Gpu::Atamic::AddNoRet(&Up, E(ii,jj,kk,EulerVIndex) * dU);
                Gpu::Atamic::AddNoRet(&Vp, E(ii,jj,kk,EulerVIndex+1) * dU);
                Gpu::Atamic::AddNoRet(&Wp, E(ii,jj,kk,EulerVIndex+2) * dU);
            }
        }
    }
}

//mParticle的成员函数
void mParticle::InteractWithEuler(MultiFab &Euler, int loop_time, Real dt, Real alpha_k, DELTA_FUNCTION_TYPE type)
{
    // for(auto it = particle_kernels.begin(); it != particle_kernels().end(); it++)
    // {
    //     InitialWithLargrangianPoints(*it)
    // }
    Vector<kernel>::iterator it;
    for(it = particle_kernels.begin();it != particle_kernels.end();it++)
    {
        InitialWithLargrangianPoin(*it);

        UpdateParticles(Euler, kernel, dt, alpha_k); //这句代码作用存疑

        const int EulerForceIndex = euler_force_index;

        while(loop_time > 0)
        {
            for(amrex::MFIter mfi(Euler); mfi.isValid(); ++mfi){
                const auto& bx = mfi.validbox();
                const auto& mf_array = Euler.array(mfi);
                amrex::ParallelFor(bx, [mf_array, EulerForceIndex] 
                AMREX_GPU_DEVICE(int i, int j, int k){
                    mf_array(i,j,k,EulerForceIndex  ) = 0.0;//+ std::exp(-r_squared);
                    mf_array(i,j,k,EulerForceIndex+1) = 0.0;
                    mf_array(i,j,k,EulerForceIndex+2) = 0.0;
                }); //对欧拉长的颗粒反作用力置为0

            }

            VelocityInterpolation(Euler, type);
            ComputeLagrangianForce(dt, kernel);
            ForceSpreading(Euler, type);

        }

}

}


//初始化颗粒
void mParticle::InitParticles(const Vector<Real>& x,
                                        const Vector<Real>& y,
                                        const Vector<Real>& z,
                                        Real rho_s,
                                        int radious)
{
    //首先检查x y z的长度是否一致，如果不一致，导入有错误
    if(x.size() == y.size() && x.size() == z.size())
    {
        Print()<<"Particle Position check is successful "<<std::endl;
    }
    else{
        return;
    }
    Real phiK = 0;
    Real h = m_gdb->Geom(euler_finest_level).CellSizeArray()[0];
    int Ml = static_cast<int>( Math::pi<Real>() / 3 * (12 * Math::powi<2>(radious / h)));
    Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * radious * radious + h * h);

    for(index = 0;index<x.size();index++)
    {
        kernel mKernel;
        mKernel.location[0] = x[index];
        mKernel.location[1] = y[index];
        mKernel.location[2] = z[index];
        mkernel.velocity[0] = 0.0;
        mkernel.velocity[1] = 0.0;
        mkernel.velocity[2] = 0.0;
        mkernel.omega[0] = 0.0;
        mkernel.omega[1] = 0.0;
        mkernel.omega[2] = 0.0;
        mKernel.varphi[0] = 0;
        mKernel.varphi[1] = 0;
        mKernel.varphi[2] = 0;
        mKernel.rho = rho_s;
        mKernel.radious = radious;
        mKernel.dv = dv;
        mKernel.ml = Ml;
        particles_kernels.push(mKernel);        
    }

    std::pair<int,int> key{0,0};
    auto& particleTileTmp = GetParticles(0)[key]; //GetParticles是保护成员函数，不能被直接调用
    if ( ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber() ) {
        //只有主进程负责初始化这些粒子，后续再通过Redistributer分发下去
        //insert particle's markers
        for(int marker_index = 0; marker_index < Ml; marker_index++){
            //insert code
            ParticleType markerP;
            markerP.id() = ParticleType::NextID();
            markerP.cpu() = ParallelDescriptor::MyProc();
            markerP.pos(0) = 0;
            markerP.pos(1) = 0;
            markerP.pos(2) = 0;

            std::array<ParticleReal, numAttri> Marker_attr;
            Marker_attr[U_Marker] = 0.0;
            Marker_attr[V_Marker] = 0.0;
            Marker_attr[W_Marker] = 0.0;
            Marker_attr[Fx_Marker] = 0.0;
            Marker_attr[Fy_Marker] = 0.0;
            Marker_attr[Fz_Marker] = 0.0;
            // attr[V] = 10.0;
            particleTileTmp.push_back(markerP);
            particleTileTmp.push_back_real(Marker_attr);
        }
    }


}


//这个函数用来生成拉格朗日点
void mParticle::InitialWithLargrangianPoints(const kernel& current_kernel){
    mParIter pti(*this, euler_finest_level);
    auto *particles  = pti.GetArrayOfStructs().data(); //返回的是颗粒结构体数组的首指针地址

    Real phiK = 0.0; //方位角

    //遍历所有颗粒，插入颗粒位置
    for(int index = 0; index < current_kernel.ml;index++)
    {
        Real Hk = -1.0 + 2.0 * (index) / ( current_kernel.ml - 1.0);
        Real thetaK = std::acos(Hk);
        if(index == 0 || index == ( current_kernel.ml - 1)){
            phiK = 0;
        }else {
            phiK = std::fmod( phiK + 3.809 / std::sqrt(current_kernel.ml) / std::sqrt( 1 - Math::powi<2>(Hk)) , 2 * Math::pi<Real>());
        }

        //这里有bug，相当于锁死了颗粒的中心在0.5 0.5 0.5这个点
        particles[index].pos(0) = 0.5 + current_kernel.radious * std::sin(thetaK) * std::cos(phiK);
        particles[index].pos(1) = 0.5 + current_kernel.radious * std::sin(thetaK) * std::sin(phiK);
        particles[index].pos(2) = 0.5 + current_kernel.radious * std::cos(thetaK);
    }
}

// void mParticle::InitialWithLargrangianPoints(const kernel& current_kernel){
//     mParIter pti(*this, euler_finest_level);
//     auto *particles = pti.GetArrayOfStructs().data();

//     Real phiK = 0;
//     for(int index = 0; index < current_kernel.ml; index++){
//         Real Hk = -1.0 + 2.0 * (index) / ( current_kernel.ml - 1.0);
//         Real thetaK = std::acos(Hk);
//         if(index == 0 || index == ( current_kernel.ml - 1)){
//             phiK = 0;
//         }else {
//             phiK = std::fmod( phiK + 3.809 / std::sqrt(current_kernel.ml) / std::sqrt( 1 - Math::powi<2>(Hk)) , 2 * Math::pi<Real>());
//         }
//         // bug here! 0.5;            
//         particles[index].pos(0) = 0.5 + current_kernel.radious * std::sin(thetaK) * std::cos(phiK);
//         particles[index].pos(1) = 0.5 + current_kernel.radious * std::sin(thetaK) * std::sin(phiK);
//         particles[index].pos(2) = 0.5 + current_kernel.radious * std::cos(thetaK);
//     }
// }



void mParticle::UpdateParticles(const amrex::MultiFab& Euler, kernel& kernel, Real dt, Real alpha_k)
{
    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    auto dxi = gm.InvCellSizeArray();

    //计算颗粒体积力，并累加到颗粒中心
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        //首先获取颗粒的信息
        auto& particles = pti.GetArrayOfStructs();
        auto *p_tr = particles.data();
        auto& attri = pti.GetAttribs();

        auto *Fxp = attri[P_attr::Fx_marker];
        auto *Fyp = attti[P_attr::Fy_marker];
        auto *Fzp = attri[P_attr::Fz_marker];
        auto *Up = attri[P_attr::U_marker];
        auto *Vp = attri[P_attr::V_marker];
        auto *Wp = attri[P_attr::W_marker];//获取所有颗粒的数据的指针，便于后期修改所有的颗粒属性值

        const Real Dv = kernel.dv;
        const Long np = pti.numParticles(); //提取其他属性

        RealVect ForceDv{std::vector<Real>{0.0,0.0,0.0}};
        RealVect Moment{std::vector<Real>{0.0,0.0,0.0}}; //力和动量

        auto *ForceDv_ptr  = &ForceDv;
        auto *Moment_ptr   = &Moment;
        auto *location_ptr = &kernel.location;
        auto *omega_ptr    = &kernel.omega;
        auto *velocity_ptr = &kernel.velocity;
        auto *varphi_ptr   = &kernel.varphi;

        const Real rho_p = kernel.rho;

        //进入一个循环，计算所有拉式点的总力
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{


            //calculate the force
            //find current particle's lagrangian marker
            //calculate the total force infect on a single particle
            //calculate the total momentum on a single particle

        });


    }
}

void mParticle::WriteParticleFile(int index)
{
    WriteAsciiFile(amrex::Concatenate("particle", index));
}



//kernels存的仍然是颗粒的质心那些数据
