#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <random>
#include "math.h"
#include "math_base.h" // don't ready
#include "tmp.h"
#include "state.h"

// Choose direction in NUTS algorithm
enum class Direction {
    Forward,
    Backward
};

// DivergenceInfo - information about integration trajectory
struct DivergenceInfo {
    std::vector<double> start_momentum;
    std::vector<double> start_location;
    std::vector<double> start_gradient;
    std::vector<double> end_location;
    double energy_error;
    int64_t end_idx_in_trajectory;
    int64_t start_idx_in_trajectory;
    std::shared_ptr<std::exception> logp_function_error; // smart pointer
};

// this structure will be created for each point
template <typename M = Math, typename P = Point>
struct LeapfrogResult {
    enum class Type { Ok, Divergence, Err }; // create enum in struct
    Type type;
    State<M, P> state;
    DivergenceInfo divergence_info;
    typename M::LogpErr error;
};


template <typename M = Math>
class Hamiltonian : public SamplerStats<M> {
public:
    using PointType = Point<M>;

    virtual LeapfrogResult<M, PointType> leapfrog(
        M& math,
        const State<M, PointType>& start,
        Direction dir,
        Collector<M, PointType>& collector
    ) = 0;

    virtual bool is_turning(
        M& math,
        const State<M, PointType>& state1,
        const State<M, PointType>& state2
    ) = 0;

    virtual State<M, PointType> init_state(
        M& math,
        const std::vector<double>& init
    ) = 0;

    virtual void initialize_trajectory(
        M& math,
        State<M, PointType>& state,
        std::mt19937& rng
    ) = 0;

    virtual StatePool<M, PointType>& pool() = 0;

    virtual State<M, PointType> copy_state(
        M& math,
        const State<M, PointType>& state
    ) = 0;

    virtual double step_size() const = 0;
    virtual void step_size(double step_size) = 0;
};

#endif // HAMILTONIAN_H