#ifndef STATE_H
#define STATE_H

#include <vector>
#include <memory>
#include <mutex>
#include "math.h"
#include "math_base.h" // don't ready
#include "tmp.h"

template <typename M = Math, typename P = Point>
class StatePool;

template <typename M = Math, typename P = Point>
class State {
public:
    State(StatePool<M, P>& pool);
    State(const State& other);
    ~State();

    const P& point() const;
    P& try_point_mut();

    int64_t index_in_trajectory() const;
    void write_position(M& math, std::vector<double>& out) const;
    void write_gradient(M& math, std::vector<double>& out) const;
    double energy() const;

private:
    std::shared_ptr<P> inner;
    StatePool<M, P>* pool;
};

template <typename M = Math, typename P = Point>
class StatePool {
public:
    StatePool(M& math, size_t capacity);
    State<M, P> new_state(M& math);
    State<M, P> copy_state(M& math, const State<M, P>& state);

private:
    std::vector<std::shared_ptr<P>> free_states;
    std::mutex mutex;
};

#endif // STATE_H