#include <iostream>
#include <vector>
#include <cmath>
#include "math.h"
#include <Eigen/Dense>

class MyTransformParams {
    
};

template <typename TransformParams>
class CpuLogpFunc {
public:
    int dim();
    double logp(
        std::vector<double> const &position,
        std::vector<double> &gradient
    );
    
    double inv_transform_normalize(
        TransformParams const &_params,
        std::vector<double> const &_untransformed_position,
        std::vector<double> const &_untransformed_gradient,
        std::vector<double> &_transformed_position,
        std::vector<double> &_transformed_gradient
    );
    
    std::pair<double,double> init_from_untransformed_position(
        TransformParams const &_params,
        std::vector<double> const &_untransformed_position,
        std::vector<double> &_untransformed_gradient,
        std::vector<double> &_transformed_position,
        std::vector<double> &_transformed_gradient
    );
    
    std::pair<double,double> init_from_transformed_position(
        TransformParams const &_params,
        std::vector<double> &_untransformed_position,
        std::vector<double> &_untransformed_gradient,
        std::vector<double> const &_transformed_position,
        std::vector<double> &_transformed_gradient
    );
    
    void update_transformation(
        // _rng - параметр для проведения рандомных вычислений
        // в C++ для этого не нужен класс
        // рандомные функции - часть стандартной библиотеки
        // пока под вопросом
        std::vector< std::vector<double> > const &_untransformed_positions,
        std::vector< std::vector<double> > const &_untransformed_gradients,
        std::vector<double> const &_untransformed_logp,
        TransformParams &_params
    );
    
    TransformParams new_transformation(
        // _rng - параметр для проведения рандомных вычислений
        // в C++ для этого не нужен класс
        // рандомные функции - часть стандартной библиотеки
        // пока под вопросом
        std::vector<double> const &_untransformed_position,
        std::vector<double> const &_untransformed_gradient,
        unsigned long long _chain
    );
    
    long long transformation_id(TransformParams const &_params);
};

class Math {
    CpuLogpFunc<MyTransformParams> logp_func;
    
    int dim() {
        return logp_func.dim();
    }
    
    Eigen::VectorXd new_array() {
        return Eigen::VectorXd::Zero(this->dim());
    }
    
    double logp_array(
        Eigen::VectorXd const &position,
        Eigen::VectorXd &gradient
    ) {
        std::vector<double> position_as_vec(
            position.data(), position.data() + position.size()
        );
        std::vector<double> gradient_as_vec(
            gradient.data(), gradient.data() + gradient.size()
        );
        logp_func.logp(position_as_vec, gradient_as_vec);
        gradient = Eigen::VectorXd::Map(
            gradient_as_vec.data(), 
            static_cast<Eigen::Index>(gradient_as_vec.size())
        );
    }
    
    double logp(
        std::vector<double> const &position,
        std::vector<double> &gradient
    ) {
        return logp_func.logp(position, gradient);
    }
};
