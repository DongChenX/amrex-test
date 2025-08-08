#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_BoxArray.H>
#include <AMReX_REAL.H>


//#include <AMReX.H>
#include <AMReX_Print.H>
//#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_MFParallelFor.H> //For the second newer method
#include <AMReX_PlotFileUtil.H> //For ploting the MultiFab
#include "ParticleTest.H"

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {

        // Number of data components at each grid point in the MultiFab
        int ncomp = 6;
        // how many grid cells in each direction over the problem domain
        int n_cell = 128;
        // how many grid cells are allowed in each direction over each box
        int max_grid_size = 128;

        // integer vector indicating the lower coordindate bounds
        amrex::IntVect dom_lo(0,0,0);
        // integer vector indicating the upper coordindate bounds
        amrex::IntVect dom_hi(n_cell-1, n_cell-1, n_cell-1);
        // box containing the coordinates of this domain
        amrex::Box domain(dom_lo, dom_hi);


        // will contain a list of boxes describing the problem domain
        amrex::BoxArray ba(domain);

        // chop the single grid into many small boxes
        ba.maxSize(max_grid_size);

        // Distribution Mapping
        amrex::DistributionMapping dm(ba);

        //Define MuliFab
        amrex::MultiFab velocity(ba, dm, ncomp, 0);

        //Geometry -- Physical Properties for data on our domain
        amrex::RealBox real_box ({0., 0., 0.}, {0.5, 0.5, 0.5});

        amrex::Geometry geom(domain, &real_box);


        // 1. 设置网格参数
        // int n_cell = 128;
        // int max_grid_size = 32;
        // int ncomp = 6; // 速度分量
        // int ngrow = 0;

        // Box domain(IntVect(0), IntVect(n_cell-1));
        // RealBox real_box({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
        // Array<int,3> is_periodic{1,1,1};

        // Geometry geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

        // BoxArray ba(domain);
        // ba.maxSize(max_grid_size);

        // DistributionMapping dm(ba);

        // 2. 创建并初始化速度场
        //MultiFab velocity(ba, dm, ncomp, ngrow);
        velocity.setVal(0.0, 0, 3); // 所有速度分量设为1
        velocity.setVal(0.0, 3, 3); // 所有速度分量设为1

        // 3. 初始化颗粒
        Real rho_f = 1.0; // 流体密度
        int force_index = 3; // 力分量起始下标
        int velocity_index = 0;
        int finest_level = 0;

        mParticle particles(geom, dm, ba, rho_f, force_index, velocity_index, finest_level);

        //颗粒参数
        Vector<Real> x{0.25}; // 球心位置
        Vector<Real> y{0.25};
        Vector<Real> z{0.25};
        Real rho_s = 20.0; // 颗粒密度
        Real radius = 0.1; // 颗粒半径（单位格点数）

        particles.InitParticles(x, y, z, rho_s, radius);

        // 4. 固定颗粒（速度为0），只做一步作用
        int loop_time = 200;
        Real dt = 0.0001;
        Real alpha_k = 0.5;
        particles.InteractWithEuler(velocity, loop_time, dt, alpha_k, DELTA_FUNCTION_TYPE::FOUR_POINT_IB);
        WriteSingleLevelPlotfile("plt001", velocity , {"ux","uy","uz","fx","fy","fz"}, geom, 0., 0);
        particles.Checkpoint("plt001","particles");
        //WriteSingleLevelPlotfile("plt001", velocity, {"comp0"}, geom, 0., 0);
    }
    amrex::Finalize();
    return 0;
}