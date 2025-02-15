#ifndef TEMP_H
#define TEMP_H

#include <memory>
#include <vector>
#include <stdexcept>
#include <arrow/array.h>
#include "state.h"
#include "math.h"
#include "math_base.h" // don't ready

// All information about point in NUTS algorithm
// from hamiltonian.rs
template <typename M = Math>
class Point {
public:
    virtual const typename M::Vector& position() const = 0;
    virtual const typename M::Vector& gradient() const = 0;
    virtual int64_t index_in_trajectory() const = 0;
    virtual double energy() const = 0;
    virtual double logp() const = 0;

    double energy_error() const {
        return energy() - initial_energy();
    }

    virtual double initial_energy() const = 0;

    virtual void copy_into(M& math, Point<M>& other) const = 0; // ?
};

// from sampler_stats.rs
// original function using class S = Settings from sample.rs file
// we will create template specialization in current files
template <typename S, typename M = Math>
class SamplerStats {
public:
    using Stats = std::vector<double>; // creating a type alias
    using Builder = std::function<void(const Stats&)>; // creating a type alias

    virtual ~SamplerStats() = default;

    // Creating builder for statistic
    virtual Builder new_builder(const S& settings, size_t dim) const = 0;

    // Get current statistic
    virtual Stats current_stats(M& math) const = 0;
};


/// Callbacks for various events during a Nuts sampling step.
///
/// Collectors can compute statistics like the mean acceptance rate
/// or collect data for mass matrix adaptation.
///
// Original trait in nuts.rs file, functions refister_draw and register_init
// will specializate in future current files
template <typename M, typename P>
class Collector {
public:
    virtual ~Collector() = default;

    // Method for step registration leapfrog
    virtual void register_leapfrog(
        M& math,
        const State<M, P>& start,
        const State<M, P>& end,
        const std::optional<DivergenceInfo>& divergence_info
    ) {}

    // virtual void register_draw(
    //     M& math,
    //     const State<M, P>& state,
    //     const SampleInfo& info
    // ) {}

    // virtual void register_init(
    //     M& math,
    //     const State<M, P>& state,
    //     const NutsOptions& options
    // ) {}
};

#endif // TEMP_H