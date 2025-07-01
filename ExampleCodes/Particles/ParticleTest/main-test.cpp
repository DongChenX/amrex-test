#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_BoxArray.H>
#include <AMReX_REAL.H>
#include "ParticleTest.H"

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        // 1. 设置网格参数
        int n_cell = 32;
        int max_grid_size = 32;
        int ncomp = 3; // 速度分量
        int ngrow = 0;

        Box domain(IntVect(0), IntVect(n_cell-1));
        RealBox real_box({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
        Array<int,3> is_periodic{1,1,1};

        Geometry geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

        BoxArray ba(domain);
        ba.maxSize(max_grid_size);

        DistributionMapping dm(ba);

        // 2. 创建并初始化速度场
        MultiFab velocity(ba, dm, ncomp, ngrow);
        velocity.setVal(1.0); // 所有速度分量设为1

        // 3. 初始化颗粒
        Real rho_f = 1.0; // 流体密度
        int force_index = 0; // 速度分量起始下标
        int velocity_index = 0;
        int finest_level = 0;

        mParticle particles(geom, dm, ba, rho_f, force_index, velocity_index, finest_level);

        // 颗粒参数
        Vector<Real> x{0.5}; // 球心位置
        Vector<Real> y{0.5};
        Vector<Real> z{0.5};
        Real rho_s = 2.0; // 颗粒密度
        int radius = 0.1 * n_cell; // 颗粒半径（单位格点数）

        particles.InitParticles(x, y, z, rho_s, radius);

        // 4. 固定颗粒（速度为0），只做一步作用
        int loop_time = 1;
        Real dt = 0.1;
        Real alpha_k = 0.5;
        particles.InteractWithEuler(velocity, loop_time, dt, alpha_k, FOUR_POINT_IB);

        // 5. 输出颗粒所受的力
        // 这里只输出第一个 kernel 的力
        const auto& kernel = particles.particle_kernels[0];
        amrex::Print() << "Particle force: "
                       << "Fx = " << kernel.velocity[0] << ", "
                       << "Fy = " << kernel.velocity[1] << ", "
                       << "Fz = " << kernel.velocity[2] << std::endl;
    }

    amrex::Finalize();
    return 0;
}