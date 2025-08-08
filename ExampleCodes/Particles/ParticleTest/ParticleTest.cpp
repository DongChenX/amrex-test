#include <AMReX_Particles.H>
#include <AMReX_MultiFabUtil.H>

#include <AMReX_RealVect.H>
#include <cmath>
#include "ParticleTest.H"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_MFParallelFor.H> //For the second newer method
#include <AMReX_PlotFileUtil.H> //For ploting the MultiFab


using namespace amrex;
using namespace std;

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
        if(rr >= 0 && rr < 1.0 )
        {
            value = 1.0 / 8.0 * ( 3.0 - 2.0 * rr + std::sqrt( 1.0 + 4.0 * rr - 4 * rr*rr) )/ h;
        }
        else if (rr >= 1.0 && rr < 2.0)
        {
            value = 1.0 / 8.0 * ( 5.0 - 2.0 * rr - std::sqrt( -7.0 + 12.0 * rr - 4 * rr*rr) )/ h;
        }
        else 
        {
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
    

    //Print()<<"get original position is "<<p.pos(0)<<" "<<p.pos(1)<<" "<<p.pos(2)<<"\n";
    //计算拉格朗日点所在的网格
    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];

    //向下取整，获取
    int index_i = static_cast<int>(std::floor(lx));
    int index_j = static_cast<int>(std::floor(ly));
    int index_k = static_cast<int>(std::floor(lz));

    //amrex::Print()<<"find the particle cell in force "<<index_i<<" "<<index_j<<" "<<index_k<<"\n";
    //index_i index_j index_k都是拉式点所在的网格
    for(int ii = index_i-2;ii<=index_i+2;ii++)
    {
        for(int jj = index_j-2;jj<=index_j+2;jj++)
        {
            for(int kk = index_k-2;kk<=index_k+2;kk++)
            {
                //计算欧拉网格的中心坐标
                Real tu,tv,tw;
                Real xf_temp = ii*dx[0] + 0.5*dx[0] + plo[0];
                Real yf_temp = jj*dx[1] + 0.5*dx[1] + plo[1];
                Real zf_temp = kk*dx[2] + 0.5*dx[2] + plo[2];

                //Print()<<"get Euler position index is "<<ii<<" "<<jj<<" "<<kk<<"\n";



                //Print()<<"get lang position index is "<<index_i<<" "<<index_j<<" "<<index_k<<"\n";

                //Print()<<"get grid data is "<<plo[0]<<" "<<plo[0]<<" "<<plo[0]<<" "<<dx[0]<<" "<<dx[1]<<" "<<dx[2]<<"\n";

                

                //Print()<<"get Euler position is "<<xf_temp<<" "<<yf_temp<<" "<<zf_temp<<"\n";

                deltaFunction(p.pos(0), xf_temp, dx[0], tu, type);
                deltaFunction(p.pos(1), yf_temp, dx[1], tv, type);
                deltaFunction(p.pos(2), zf_temp, dx[2], tw, type);
                //Print()<<"get interplation data is "<<tu<<" "<<tv<<" "<<tw<<"\n";
                Real dU = tu*tv*tw*d;
                //Print()<<"get position is "<<p.pos(0)<<" "<<p.pos(1)<<" "<<p.pos(2)<<"\n";



                //Print()<<"get force is "<<tu<<" "<<tv<<" "<<tw<<" "<<dU<<"\n";
                //Print()<<"get force is "<<fxP*dU<<" "<<fyP*dU<<" "<<fzP*dU<<"\n";
                //Print()<<"get computed force is "<<fxP<<" "<<fyP<<" "<<fzP<<"\n";
                //Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex), fxP*dU);
                E(ii,jj,kk,EulerFIndex) += fxP*dU;
                E(ii,jj,kk,EulerFIndex+1) += fyP*dU;
                E(ii,jj,kk,EulerFIndex+2) += fzP*dU;
                //amrex::Print()<<"Feild spread fore is "<<EulerFIndex<<" "<<E(ii,jj,kk,EulerFIndex)<<" "<<E(ii,jj,kk,EulerFIndex+1)<<" "<<E(ii,jj,kk,EulerFIndex+2)<<"\n";
                //Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex+1), fyP*dU);
                //Gpu::Atomic::AddNoRet(&E(ii,jj,kk,EulerFIndex+2), fzP*dU);
            }
        }
    }
    
}


//这里是将欧式点的速度插值到拉式点上面
template <typename P = Particle<numAttr> >
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void VelocityInterpolation_cir(P const& p, Real& Up, Real& Vp, Real& Wp,
                     Array4<Real const> const& E, int EulerVIndex,
                     GpuArray<Real, AMREX_SPACEDIM> const& plo,
                     GpuArray<Real, AMREX_SPACEDIM> const& dx,
                     DELTA_FUNCTION_TYPE type)
{
    const Real d = dx[0]*dx[1]*dx[2]; //计算欧拉网格的体积
    //amrex::Print()<<"FILED LOW "<<plo[0]<<" "<<plo[1]<<" "<<plo[2]<<"\n";
    //amrex::Print()<<"FILED dx "<<dx[0]<<" "<<dx[1]<<" "<<dx[2]<<"\n";

    Real lx = (p.pos(0) - plo[0]) / dx[0];
    Real ly = (p.pos(1) - plo[1]) / dx[1];
    Real lz = (p.pos(2) - plo[2]) / dx[2];
    //amrex::Print()<<"find the particle lenth "<<lx<<" "<<ly<<" "<<lz<<"\n";

    //向下取整，获取
    int index_i = static_cast<int>(std::floor(lx));
    int index_j = static_cast<int>(std::floor(ly));
    int index_k = static_cast<int>(std::floor(lz));
    //amrex::Print()<<"find the particle cell "<<index_i<<" "<<index_j<<" "<<index_k<<"\n";

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
                Real xf_temp = ii*dx[0] + 0.5*dx[0] + plo[0];
                Real yf_temp = jj*dx[1] + 0.5*dx[1] + plo[1];
                Real zf_temp = kk*dx[2] + 0.5*dx[2] + plo[2];
                //deltaFunction(Real xf, Real xp, Real h, Real& value, DELTA_FUNCTION_TYPE type)
                deltaFunction(p.pos(0), xf_temp, dx[0], tu, type);
                deltaFunction(p.pos(1), yf_temp, dx[1], tv, type);
                deltaFunction(p.pos(2), zf_temp, dx[2], tw, type);
                Real dU = tu*tv*tw*d;
                Up += E(ii,jj,kk,EulerVIndex) * dU;
                Vp += E(ii,jj,kk,EulerVIndex+1) * dU;
                Wp += E(ii,jj,kk,EulerVIndex+2) * dU;
                //amrex::Print()<<"Euler cell is  "<<ii<<" "<<jj<<" "<<kk<<"\n";
                //amrex::Print()<<"laange velocity is  "<<Up<<" "<<Vp<<" "<<Wp<<"\n";
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
    const auto& gm = m_gdb->Geom(euler_finest_level);
    Vector<kernel>::iterator it;
    for(it = particle_kernels.begin();it != particle_kernels.end();it++)
    {
        InitialWithLargrangianPoints(*it);

        // WritePlotFile("plts001", "particle");

        // UpdateParticles(Euler, *it, dt, alpha_k);  //更新所有拉氏点的数据，位置和速度

        const int EulerForceIndex = euler_force_index;
        int index_number = 0;
        while(loop_time > 0)
        {
            for(amrex::MFIter mfi(Euler); mfi.isValid(); ++mfi){
                const auto& bx = mfi.validbox();
                const auto& mf_array = Euler.array(mfi);
                amrex::ParallelFor(bx, [mf_array, EulerForceIndex] 
                AMREX_GPU_DEVICE(int i, int j, int k){
                    mf_array(i,j,k,EulerForceIndex  ) = 0.0;//
                    mf_array(i,j,k,EulerForceIndex+1) = 0.0;
                    mf_array(i,j,k,EulerForceIndex+2) = 0.0;
                }); //对欧拉长的颗粒反作用力置为0
            }
            VelocityInterpolation(Euler,type);
            WriteParticledata(index_number++);
            ComputeLagrangianForce(dt, *it, 0);
            ForceSpreading(Euler, type);
            for(amrex::MFIter mfi(Euler); mfi.isValid(); ++mfi){
                const amrex::Box& bx = mfi.validbox();
                const auto& mf_array = Euler[mfi].array();
                    //Print()<<"Euler FORCE INDEX IS "<<euler_force_index<<" "<<euler_force_index+2<<"\n";
                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
                    mf_array(i,j,k,euler_velocity_index) = mf_array(i,j,k,euler_velocity_index) + dt * mf_array(i,j,k,euler_force_index);

                    mf_array(i,j,k,euler_velocity_index+1) = mf_array(i,j,k,euler_velocity_index+1) + dt * mf_array(i,j,k,euler_force_index+1);
                    mf_array(i,j,k,euler_velocity_index + 2) = mf_array(i,j,k,euler_velocity_index+2) + dt * mf_array(i,j,k,euler_force_index+2);
                });
            }
            Print()<<"loop over the grid is over"<<"\n";    
            //MultiFab::Saxpy(Euler, dt, Euler, euler_force_index, euler_velocity_index, 3, 0);
            
            loop_time --;
        }

    }

}


//初始化颗粒
void mParticle::InitParticles(const Vector<Real>& x,
                                        const Vector<Real>& y,
                                        const Vector<Real>& z,
                                        Real rho_s,
                                        Real radious)
{
    //首先检查x y z的长度是否一致，如果不一致，导入有错误
    if(x.size() == y.size() && x.size() == z.size())
    {
        Print()<<"Particle Position check is successful "<<std::endl;
    }
    else{
        return;
    }
    //  Real h = m_gdb->Geom(euler_finest_level).CellSizeArray()[0];
    // int Ml = static_cast<int>( Math::pi<Real>() / 3 * (12 * Math::powi<2>(radious / h)));
    // Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * radious * radious + h * h);

    Real h = m_gdb->Geom(euler_finest_level).CellSizeArray()[0];
    //bug is here
    //int Ml = static_cast<int>( Math::pi<Real>() / 3 * (12 * Math::powi<2>(radious / h)));

    int Ml = static_cast<int> (PI/3.0 *(12.0*(radious/h)*(radious/h) + 1.0));
    Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * radious * radious + h * h);


    //Print()<<h<<" "<<Ml<<"\n";
    //Real dv = Math::pi<Real>() * h / 3 / Ml * (12 * radious * radious + h * h);
    //Print() << dv <<"\n";
    for(int index = 0; index<x.size(); index++)
    {
        kernel mKernel;
        mKernel.location[0] = x[index];
        mKernel.location[1] = y[index];
        mKernel.location[2] = z[index];
        mKernel.velocity[0] = 0.0;
        mKernel.velocity[1] = 0.0;
        mKernel.velocity[2] = 0.0;
        mKernel.omega[0] = 0.0;
        mKernel.omega[1] = 0.0;
        mKernel.omega[2] = 0.0;
        mKernel.varphi[0] = 0;
        mKernel.varphi[1] = 0;
        mKernel.varphi[2] = 0;
        mKernel.rho = rho_s;
        mKernel.radious = radious;
        mKernel.dv = dv;
        mKernel.ml = Ml;
        particle_kernels.push_back(mKernel);        
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

            std::array<ParticleReal, numAttr> Marker_attr;
            Marker_attr[U_marker] = 0.0;
            Marker_attr[V_marker] = 0.0;
            Marker_attr[W_marker] = 0.0;
            Marker_attr[Fx_marker] = 0.0;
            Marker_attr[Fy_marker] = 0.0;
            Marker_attr[Fz_marker] = 0.0;
            // attr[V] = 10.0;
            particleTileTmp.push_back(markerP);
            particleTileTmp.push_back_real(Marker_attr);
        }
    }
    Redistribute();
}


//这个函数用来生成拉格朗日点
//bug is here
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

        //这里有bug，相当于锁死了颗粒的中心在0.5 0.5 0.5这个点, 应该做修改
        particles[index].pos(0) = current_kernel.location[0] + current_kernel.radious * std::sin(thetaK) * std::cos(phiK);
        particles[index].pos(1) = current_kernel.location[1] + current_kernel.radious * std::sin(thetaK) * std::sin(phiK);
        particles[index].pos(2) = current_kernel.location[2] + current_kernel.radious * std::cos(thetaK);
        //Print()<<"The index of particle and position "<<index << " "<< particles[index].pos(0) <<" "<<particles[index].pos(1)<<" "<<particles[index].pos(2)<<"\n";
    }
}



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

        auto *Fxp = attri[P_attr::Fx_marker].data();
        auto *Fyp = attri[P_attr::Fy_marker].data();
        auto *Fzp = attri[P_attr::Fz_marker].data();
        auto *Up = attri[P_attr::U_marker].data();
        auto *Vp = attri[P_attr::V_marker].data();
        auto *Wp = attri[P_attr::W_marker].data();//获取所有颗粒的数据的指针，便于后期修改所有的颗粒属性值

        const Real Dv = kernel.dv;
        const Long np = pti.numParticles(); //提取每一个tile中的颗粒的数量

        RealVect ForceDv{std::vector<Real>{0.0,0.0,0.0}};
        RealVect Moment{std::vector<Real>{0.0,0.0,0.0}}; //力和动量


        auto *ForceDv_ptr  = &ForceDv;
        auto *Moment_ptr   = &Moment;
        auto *location_ptr = &kernel.location;
        auto *omega_ptr    = &kernel.omega;
        auto *velocity_ptr = &kernel.velocity;
        auto *varphi_ptr   = &kernel.varphi;

        const Real rho_p = kernel.rho;
        const Real rho_f = (this->euler_fluid_rho);
        const Real radious_p = kernel.radious;
        Real particleVolume = 4.0/3.0*PI*radious_p*radious_p*radious_p;

        //进入一个循环，计算所有拉式点的总力
        //AMREX_D_DEL
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            //初始化一个向量，计算每一个拉氏点的力，并且把这个力写入到质心坐标中
            RealVect Force_temp(AMREX_D_DECL(Fxp[i], Fyp[i], Fzp[i])); //这里的FX,FY,FZ都是力的密度，不是合力
            *ForceDv_ptr += Force_temp*Dv;
            //RealVect R_temp(AMREX_D_DECL(p_tr[i].pos(0) - location_ptr[0], p_tr[i].pos(1) - location_ptr[1], p_tr[i].pos(2) - location_ptr[2])); // 计算拉氏点与颗粒中心的矢径
            // Vector<Real> data_temp{p_tr[i].pos(0) - location_ptr[0], 
            //                  p_tr[i].pos(1) - location_ptr[1], 
            //                  p_tr[i].pos(2) - location_ptr[2]};
            // RealVect R_temp(data_temp);
                RealVect R_temp(RealVect(p_tr[i].pos(0), p_tr[i].pos(1), p_tr[i].pos(2)) - *location_ptr);
            RealVect mom_temp = R_temp.crossProduct(Force_temp)*Dv;
            *Moment_ptr += mom_temp;
        });

        //计算颗粒的速度
        auto oldParticleVelocity = *velocity_ptr;
        auto oldParticleomega = *omega_ptr;
        RealVect newParticleVelocity = oldParticleVelocity - 2*alpha_k *(*ForceDv_ptr) *dt/(rho_p - rho_f)/particleVolume;
        RealVect newParticleomega = oldParticleomega - 2*alpha_k*(*Moment_ptr)*dt/(rho_p - rho_f)/particleVolume;
        

        //更新颗粒物的位置
        // auto deltaX = alpha_k*dt*(oldParticleVelocity + newParticleVelocity) / 2.0;
        // *location_ptr = *location_ptr + deltaX;
        // *varphi_ptr = *varphi_ptr + alpha_k*dt*(newParticleomega + oldParticleomega) / 2.0;

        //更新所有拉氏点的速度
        // amrex::ParallelFor(np, [=] 
        // AMREX_GPU_DEVICE (int i) noexcept{
        //     //初始化一个向量，计算每一个拉氏点的力，并且把这个力写入到质心坐标中
        //     // RealVect Force_temp{Fxp[i], Fyp[i], Fzp[i]}; //这里的FX,FY,FZ都是力的密度，不是合力
        //     // *ForceDv_ptr += Force_temp*Dv;
        //     // RealVect R_temp = std::vector{p_tr[i].pos(0) - location_ptr[0], p_tr[i].pos(1) - location_ptr[1], p_tr[i].pos(k) - location_ptr[2]}; // 计算拉氏点与颗粒中心的矢径
        //     // RealVect mom_temp = R_temp.crossProduct(Force_temp)*Dv;
        //     // *Moment_ptr += mom_temp;
        //     p_tr[i].pos(0) += 
        // });
    }
}

void mParticle::VelocityInterpolation(const amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type)
{
    //amrex::Print()<<"Now is the interplation"<<"\n";
    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    //auto dxi = gm.InvCellSizeArray();
    amrex::GpuArray<amrex::Real,3> dxi = gm.CellSizeArray(); 
    
    //计算颗粒体积力，并累加到颗粒中心
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        //首先获取颗粒的信息
        auto& particles = pti.GetArrayOfStructs();
        auto *p_tr = particles.data();
        auto& attri = pti.GetAttribs();

        auto *Up = attri[P_attr::U_marker].data();
        auto *Vp = attri[P_attr::V_marker].data();
        auto *Wp = attri[P_attr::W_marker].data();//获取所有颗粒的数据的指针，便于后期修改所有的颗粒属性值

        const Long np = pti.numParticles(); //提取每一个tile中的颗粒的数量


        //进入一个循环，计算所有拉式点的总力
        //AMREX_D_DEL
        const auto& E = Euler[pti].array();
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            VelocityInterpolation_cir(p_tr[i], Up[i],Vp[i],Wp[i], E, euler_velocity_index, plo, dxi, type);
            //Print()<<"lagerange particle index is "<<i<<"\n";
        });
    }
    //WriteAsciiFile(amrex::Concatenate("particle", 1));
}

void mParticle::ForceSpreading(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type)
{
    const auto& gm = m_gdb->Geom(euler_finest_level);
    auto plo = gm.ProbLoArray();
    amrex::GpuArray<amrex::Real,3> dxi = gm.CellSizeArray(); 

    
    //计算颗粒体积力，并累加到颗粒中心
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        //首先获取颗粒的信息
        auto& particles = pti.GetArrayOfStructs();
        auto *p_tr = particles.data();
        auto& attri = pti.GetAttribs();

        auto * const Fxp = attri[P_attr::Fx_marker].data();
        auto * const Fyp= attri[P_attr::Fy_marker].data();
        auto * const Fzp = attri[P_attr::Fz_marker].data();

        // const Real Dv = kernel.dv;
        const Long np = pti.numParticles(); //提取每一个tile中的颗粒的数量

        const auto& E = Euler[pti].array();
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
            ForceSpreading_cic(p_tr[i], Fxp[i], Fyp[i], Fzp[i], E, euler_force_index, plo, dxi, type);
        });
    }
    //WriteAsciiFile(amrex::Concatenate("particle", 1));
}

void mParticle::ComputeLagrangianForce(Real dt, const kernel& kernel,int index){

    // Real Up = kernel.velocity[0];
    // Real Vp = kernel.velocity[1];
    // Real Wp = kernel.velocity[2];
    Real Upm = 1.0;
    Real Vpm = 1.0;
    Real Wpm = 1.0;
    

    //amrex::Print()<<"Now is the compute force"<<"\n";
    //如果把颗粒的速度卡死，经过多次重复迭代，可以知道颗粒周围流场也是0，以满足无滑移边界条件

    //计算颗粒体积力，并累加到颗粒中心
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti){
        //首先获取颗粒的信息
        auto& particles = pti.GetArrayOfStructs();
        auto *p_tr = particles.data();
        auto& attri = pti.GetAttribs();

        auto *Up = attri[P_attr::U_marker].data();
        auto *Vp = attri[P_attr::V_marker].data();
        auto *Wp = attri[P_attr::W_marker].data();
        //amrex::Print()<<"The vel data is "<<Up[60]<<" "<<Vp[60]<< " "<<Wp[60]<<"\n";

        auto *Fxp = attri[P_attr::Fx_marker].data();
        auto *Fyp = attri[P_attr::Fy_marker].data();
        auto *Fzp = attri[P_attr::Fz_marker].data();

        const Long np = pti.numParticles(); //提取每一个tile中的颗粒的数量
        //amrex::Print()<<"the particle number is "<<np<<"\n";
        //mrex::Array4<amrex::Real>& E = Euler[pti].array();
        amrex::ParallelFor(np, [=] 
        AMREX_GPU_DEVICE (int i) noexcept{
        //amrex::Print()<<"The langaragian velocity is "<<Fxp[100]<<" "<<Fyp[100]<< " "<<Fzp[100]<<"\n";
            Fxp[i] = (Upm - Up[i])/dt;
            Fyp[i] = (Vpm - Vp[i])/dt;
            Fzp[i] = (Wpm - Wp[i])/dt;
        });
        amrex::Print()<<"The calculated force of point_100:  "<<Fxp[100]<<" "<<Fyp[100]<< " "<<Fzp[100]<<"\n";
        amrex::Print()<<"The velocity of lagrangian point_100: "<<Up[100]<<" "<<Vp[100]<< " "<<Wp[100]<<"\n";
        amrex::Print()<<"The velocity of lagrangian points: "<<Up[100]<<" "<<Vp[50]<< " "<<Wp[200]<<"\n";
    }
}


void mParticle::WriteParticleFile(int index)
{
    WriteAsciiFile(amrex::Concatenate("particle", index));
}


void mParticle::WriteParticledata(int index)
{
    std::string filename = "particles_step" + std::to_string(index) + ".vtk";
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // VTK头
    file << "# vtk DataFile Version 3.0\n";
    file << "Particle Data\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";


    Particledata p_temp;
    for(mParIter pti(*this, euler_finest_level); pti.isValid(); ++pti)
    {
        auto* p_tr = pti.GetArrayOfStructs().data();
        auto& attri = pti.GetAttribs();
        auto* Up = attri[P_attr::U_marker].data();
        auto* Vp = attri[P_attr::V_marker].data();
        auto* Wp = attri[P_attr::W_marker].data();
        auto* Fxp = attri[P_attr::Fx_marker].data();
        auto* Fyp = attri[P_attr::Fy_marker].data();
        auto* Fzp = attri[P_attr::Fz_marker].data();
        long int np = pti.numParticles();
        for(long int i  = 0; i < np; i++)
        {
            p_temp.particleX.push_back(p_tr[i].pos(0));
            p_temp.particleY.push_back(p_tr[i].pos(1));
            p_temp.particleZ.push_back(p_tr[i].pos(2));
            p_temp.particleU.push_back(Up[i]);
            p_temp.particleV.push_back(Vp[i]);
            p_temp.particleW.push_back(Wp[i]);
            p_temp.particleFx.push_back(Fxp[i]);
            p_temp.particleFy.push_back(Fyp[i]);
            p_temp.particleFz.push_back(Fzp[i]);
        }
    }
    long int particleSize = p_temp.particleX.size();
    
    //写入点坐标
    file << "POINTS " << particleSize << " double\n";
    for (long int i = 0; i<particleSize; i++) {
        file << std::fixed << std::setprecision(6) 
                << p_temp.particleX[i] << " " << p_temp.particleY[i] << " " << p_temp.particleZ[i] << "\n";
    }
    file << "\n";

    //写入单元格（每个颗粒作为一个顶点单元格）
    file<<"CELLS "<<particleSize<<" "<<particleSize*2<<"\n";
    for(long int i = 0; i<particleSize; i++)
    {
        file <<"1 "<< i <<"\n";
    }
    file<<"\n";

    // 写入单元格类型（1表示顶点）
    file << "CELL_TYPES " << particleSize<< "\n";
    for (size_t i = 0; i < particleSize; ++i) {
        file << "1\n";
    }
    file << "\n";

    //写入点的数据
    file << "POINT_DATA "<<particleSize<<"\n";

    //写入速度
    file <<"VECTORS velocity double \n";
    for (long int i = 0; i<particleSize; i++) {
        file << std::fixed << std::setprecision(6) 
                << p_temp.particleU[i] << " " << p_temp.particleV[i] << " " << p_temp.particleW[i] << "\n";
    }

    file <<"\n";

    //写入迭代过程中的作用力数据
    file <<"VECTORS force double \n";
    for (long int i = 0; i<particleSize; i++) {
        file << std::fixed << std::setprecision(6) 
                << p_temp.particleFx[i] << " " << p_temp.particleFy[i] << " " << p_temp.particleFz[i] << "\n";
    }
    file << "\n";
}
