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
struct kernel{
    RealVect velocity;
    RealVect location;
    RealVect omega; //颗粒的角速度
    RealVect varphi; //颗粒的累积角位移
    Real rho;
    Real radious;
    Real ml; //拉式点的总数
    Real dv; //每个marker代表的积分权重，拉式点的体积
};


//定义一个拉氏点所需要的枚举类型
enum P_attr{
    U_marker = 0,
    V_marker,
    W_marker,
    Fx_marker,
    Fy_marker,
    Fz_marker,
    numAttr
};

//定义一个类的迭代器，可以查看颗粒的数据
class mParIter : public ParIter<0,0,numAttr>
{
    public:
        using amrex::ParIter<0,0,numAttr>::ParIter;
        using RealVector = amrex::ParIter<0,0,numAttr>::ContainerType::RealVector;
        [[nodiscard]] const std::array<RealVector, numAttr>& GetAttribs () const
        {
            return GetStructOfArrays().GetRealData();
        }

        [[nodiscard]] const RealVector& GetAttribs(int comp) const{
            return GetStructOfArrays().GetRealData(comp); //comp是拉氏点的坐标
        }

        std::array<RealVector, numAttr>& GetAttribs()
        {
            return GetStructOfArrays().GetRealData();
        }

        RealVector& GetAttribs(int comp)
        {
            return GetStructOfArrays().GetRealData(comp);
        }
};


//定义delta function
void deltafunction(Real xf, Real xp, Real h, Real &value); //xf为欧氏点的坐标， xp为拉氏点的坐标，h为网格间距，value为具体的物理量

//定义一个颗粒容器，存储一个颗粒所有的拉氏点
class mParticle : public ParticleContainer<0, 0, numAttr>{
    //定义构造函数
    public:
    explicit mParticle(
        const Vector<Geometry>   &gm,
        const Vector<DistributionMapping>  &dm,
        const Vector<BoxArray>   &ba,
        const Vector<int>        &rr
    ):ParticleContainer<0, 0, numAttr>(gm, dm, ba, rr){}

    explicit mParticle(
        const Geometry&  gm,
        const DistributionMapping&  dm,
        const BoxArray&  ba,
        Real rho_f,
        int force_index,
        int velocity_index,
        int finest_level
    ): ParticleContainer<0, 0, numAttr>(gm, dm, ba), euler_fluid_rho(rho_f), euler_force_index(force_index), euler_velocity_index(velocity_index), euler_finest_level(finest_level){}

    void InitParticles(const Vector<Real>& x,
                       const Vector<Real>& y,
                       const Vector<Real>& z,
                       Real rho_s,
                       int radious);  //x, y, z都是拉氏点的坐标， rho_s是颗粒的密度， radius是颗粒的半径，这些是计算颗粒表面拉氏点的关键位置

    void InteractWithEuler(MultiFab &Euler, int loop_time = 20, Real dt = 0.1, Real alpha_k = 0.5, DELTA_FUNCTION_TYPE type = FOUR_POINT_IB); //将颗粒边界上的信息插值到附近的拉氏点上面

    void WriteParticleFile(int index); //写入颗粒的信息


    private:

        int euler_finest_level; //最高层级的网格，网格最精细

        int euler_force_index;  //欧拉力的索引

        int euler_velocity_index; //欧拉点的速度索引

        Real euler_fluid_rho; //流体的密度

        Vector<kernel> particle_kernels; //质心的关键物理量

        //整体的数据结构就是颗粒质心存多个，但是拉氏点的数据每一次都换一下重新计算一下数据
        void InitialWithLargrangianPoints(const kernel& current_kernel); //初始化拉氏点的信息

        void VelocityInterpolation(const amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type); //将拉氏点的速度插值到附近的欧拉点上面

        void ForceSpreading(amrex::MultiFab &Euler, DELTA_FUNCTION_TYPE type);

        void UpdateParticles(const amrex::MultiFab& Euler, kernel& kernel, Real dt, Real alpha_k ); //更新颗粒物的信息
        
        void ComputeLagrangianForce(Real dt, const kernel& kernel); //计算拉氏力

};
#endif // 