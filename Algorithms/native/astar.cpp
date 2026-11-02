// Exact 8/26-connected A*. No fast-math, heuristic weighting or smoothing.
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>
namespace py = pybind11;
using I = std::int64_t;
using Point = std::array<I,3>;
struct Move { Point delta; I offset; double cost; std::vector<I> checks; };
struct Entry { double f, g; I id; };
struct Order {
    bool operator()(const Entry& a, const Entry& b) const {
        if (a.f != b.f) return a.f > b.f;
        if (a.g != b.g) return a.g < b.g;
        return a.id > b.id;
    }
};
py::tuple search(py::array_t<std::uint8_t, py::array::c_style> grid,
                 std::vector<I> start, std::vector<I> end,
                 double timeout, bool corners) {
    auto info = grid.request();
    const int nd = info.ndim;
    if ((nd != 2 && nd != 3) || start.size()!=size_t(nd) || end.size()!=size_t(nd))
        throw std::invalid_argument("A* needs 2D/3D occupancy and matching coordinates");
    if (!std::isfinite(timeout) || timeout < 0) throw std::invalid_argument("invalid timeout");
    const auto* input = static_cast<const std::uint8_t*>(info.ptr);
    for (const auto& pt : {start,end}) {
        I idx=0;
        for (int d=0;d<nd;++d) {
            if (pt[d]<0 || pt[d]>=info.shape[d]) throw std::invalid_argument("endpoint outside grid");
            idx=idx*info.shape[d]+pt[d];
        }
        if(input[idx]) throw std::invalid_argument("blocked endpoint");
    }
    std::vector<std::vector<I>> path;
    I expanded=0,popped=0,stale=0,pushes=1,peak=1;
    double path_cost=0;
    std::string code="unreachable";
    {
        py::gil_scoped_release release;
        const auto began=std::chrono::steady_clock::now();
        Point shape={1,1,1}, stride={1,1,1}, goal={0,0,0};
        for(int d=0;d<nd;++d) { shape[d]=info.shape[d]+2; goal[d]=end[d]+1; }
        for(int d=nd-2;d>=0;--d) stride[d]=stride[d+1]*shape[d+1];
        I size=shape[0]*stride[0], s=0,e=0;
        for(int d=0;d<nd;++d) {s+=(start[d]+1)*stride[d];e+=goal[d]*stride[d];}
        std::vector<std::uint8_t> blocked(size,1), closed(size,0);
        for(I i=0;i<info.size;++i) {
            I rem=i,idx=0;
            for(int d=nd-1;d>=0;--d) { idx+=(rem%info.shape[d]+1)*stride[d];rem/=info.shape[d]; }
            blocked[idx]=input[i]!=0;
        }
        std::vector<Move> moves;
        for(int a=-1;a<=1;++a) for(int b=-1;b<=1;++b)
        for(int c=(nd==3?-1:0);c<=(nd==3?1:0);++c) {
            Point delta={a,b,c}; int axes=a*a+b*b+c*c;
            if(!axes) continue;
            Move m{delta, a*stride[0]+b*stride[1]+c*stride[2], std::sqrt(double(axes)),{}};
            if(!corners) {
                std::vector<I> active;
                for(int d=0;d<nd;++d) if(delta[d]) active.push_back(delta[d]*stride[d]);
                for(unsigned bits=1;bits+1<(1u<<active.size());++bits) {
                    I off=0;for(unsigned j=0;j<active.size();++j) if(bits&(1u<<j)) off+=active[j];
                    m.checks.push_back(off);
                }
            }
            moves.push_back(m);
        }
        const double sq2=std::sqrt(2.),sq3=std::sqrt(3.);
        auto h=[&](I id) {
            Point dist={0,0,0};
            for(int d=0;d<nd;++d) {dist[d]=std::abs(id/stride[d]-goal[d]);id%=stride[d];}
            if(nd==2) return double(std::max(dist[0],dist[1]))+(sq2-1)*std::min(dist[0],dist[1]);
            std::sort(dist.begin(),dist.end());
            return double(dist[2])+(sq2-1)*dist[1]+(sq3-sq2)*dist[0];
        };
        std::vector<double> costs(size,std::numeric_limits<double>::infinity());
        std::vector<I> prev(size,-1);
        std::priority_queue<Entry,std::vector<Entry>,Order> queue;
        queue.push({h(s),0,s});costs[s]=0;
        while(!queue.empty()) {
            // Check even for zero timeout, then every 256 pops.
            if ((popped & 255)==0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()>=timeout) {
                code="search_timeout";break;
            }
            auto top=queue.top();queue.pop();++popped;
            I cur=top.id;
            if(closed[cur] || top.g>costs[cur]) {++stale;continue;}
            if(cur==e) {
                path_cost=top.g;code.clear();
                for(;cur!=-1;cur=prev[cur]) {
                    I rem=cur;std::vector<I> pt(nd);
                    for(int d=0;d<nd;++d) {pt[nd-1-d]=rem/stride[d]-1;rem%=stride[d];}
                    path.push_back(pt);
                }
                std::reverse(path.begin(),path.end());break;
            }
            closed[cur]=1;++expanded;
            for(const auto& move:moves) {
                I nb=cur+move.offset;
                if(blocked[nb] || closed[nb]) continue;
                double ng=top.g+move.cost;
                if(ng>=costs[nb]) continue;
                bool hit=false;
                for(I offset:move.checks) if(blocked[cur+offset]) {hit=true;break;}
                if(hit) continue;
                costs[nb]=ng;prev[nb]=cur;
                queue.push({ng+h(nb),ng,nb});++pushes;
            }
            peak=std::max(peak,I(queue.size()));
        }
    }
    py::dict stats;
    stats["expanded"]=expanded;stats["popped"]=popped;stats["stale_pops"]=stale;
    stats["pushes"]=pushes;stats["peak_open"]=peak;
    stats["path_cost"]=code.empty()?py::cast(path_cost):py::none();
    return py::make_tuple(code.empty()?py::cast(path):py::none(),
                          code.empty()?py::none():py::cast(code),stats);
}
PYBIND11_MODULE(_astar_native,m) { m.def("search",&search); }
