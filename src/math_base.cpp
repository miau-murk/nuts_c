#include <iostream>
#include <vector>
#include <cmath>
#include "math.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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
public:
    int dim() {
        return logp_func.dim();
    }
    
    Eigen::VectorXd new_array() {
        return Eigen::VectorXd::Zero(this->dim());
    }
    
    // new_eig_vectors
    
    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType 
        new_eig_values(std::vector<double> vals) 
    {
        Eigen::VectorXd values = Eigen::VectorXd::Map(
            vals.data(), static_cast<Eigen::Index>(vals.size())
        );
        return values;
    }
    
    double logp_array(
        Eigen::VectorXd const &position,
        Eigen::VectorXd &gradient
    ) {
        std::vector<double> position_as_vec(
            position.data(), position.data() + position.size()
        );        std::vector<double> gradient_as_vec(
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
    
    std::pair<double,double> scalar_prods2(
        Eigen::VectorXd const &positive1,
        Eigen::VectorXd const &positive2,
        Eigen::VectorXd const &x,
        Eigen::VectorXd const &y
    ) 
    {
        std::vector<double> vx(x.data(), x.data()+x.size());
        std::vector<double> vy(y.data(), y.data()+y.size());
        std::vector<double> vpositive1(
            positive1.data(), positive1.data() + positive1.size()
        );
        std::vector<double> vpositive2(
            positive2.data(), positive2.data() + positive2.size()
        );
        return myMath::scalar_prods2(
            vpositive1, vpositive2, vx, vy
        );
    }
    
    std::pair<double,double> scalar_prods3(
        Eigen::VectorXd const &positive1,
        Eigen::VectorXd const &negative1,
        Eigen::VectorXd const &positive2,
        Eigen::VectorXd const &x,
        Eigen::VectorXd const &y
    ) 
    {
        std::vector<double> vx(x.data(), x.data()+x.size());
        std::vector<double> vy(y.data(), y.data()+y.size());
        std::vector<double> vpositive1(
            positive1.data(), positive1.data() + positive1.size()
        );
        std::vector<double> vnegative1(
            negative1.data(), negative1.data() + negative1.size()
        );
        std::vector<double> vpositive2(
            positive2.data(), positive2.data() + positive2.size()
        );
        return myMath::scalar_prods3(
            vpositive1, vnegative1, vpositive2, vx, vy
        );
    }
    
    void axpy_out(
        Eigen::VectorXd const &x,
        Eigen::VectorXd const &y,
        double a,
        Eigen::VectorXd &out
    ) {
        std::vector<double> vx(x.data(), x.data()+x.size());
        std::vector<double> vy(y.data(), y.data()+y.size());
        std::vector<double> vout(
            out.data(), out.data() + out.size()
        );
        myMath::axpy_out(vx, vy, a, vout);
        out = Eigen::VectorXd::Map(
            vout.data(), static_cast<Eigen::Index>(vout.size())
        );
    }
    
    void axpy(
        Eigen::VectorXd const &x,
        Eigen::VectorXd &y,
        double a
    ) {
        std::vector<double> vx(x.data(), x.data()+x.size());
        std::vector<double> vy(y.data(), y.data()+y.size());
        myMath::axpy(vx, vy, a);
        y = Eigen::VectorXd::Map(
            vy.data(), static_cast<Eigen::Index>(vy.size())
        );
    }
    
    double sq_norm_sum(
        Eigen::VectorXd const &x,
        Eigen::VectorXd const &y
    ) {
        std::vector<double> vx(x.data(), x.data()+x.size());
        std::vector<double> vy(y.data(), y.data()+y.size());
        double sm = 0;
        for (int i = 0; i < (int) vx.size(); i++) {
            sm += (vx[i] + vy[i]) * (vx[i] + vy[i]);
        }
        return sm;
    }
    
    void read_from_slice(
        Eigen::VectorXd &dest,
        std::vector<double> const &source
    ) {
        dest = Eigen::VectorXd::Map(
            source.data(), 
            static_cast<Eigen::Index>(source.size())
        );
    }
    
    void write_to_slice(
        Eigen::VectorXd const &source,
        std::vector<double> &dest
    ) {
        dest = std::vector<double>(
            source.data(), source.data() + source.size()
        );
    }
    
    void copy_into(
        Eigen::VectorXd const &array,
        Eigen::VectorXd &dest
    ) {
        dest = Eigen::VectorXd::Map(
            array.data(), 
            static_cast<Eigen::Index>(array.size())
        );
    }
    
    void array_mult(
        Eigen::VectorXd const &array1,
        Eigen::VectorXd const &array2,
        Eigen::VectorXd &dest
    ) {
        std::vector<double> varray1(
            array1.data(), array1.data() + array1.size()
        );
        std::vector<double> varray2(
            array2.data(), array2.data() + array2.size()
        );
        std::vector<double> vdest(
            dest.data(), dest.data() + dest.size()
        );
        myMath::multiply(varray1, varray2, vdest);
        dest = Eigen::VectorXd::Map(
            vdest.data(), 
            static_cast<Eigen::Index>(vdest.size())
        );
    }
    
    double array_vector_dot(
        Eigen::VectorXd const &array1,
        Eigen::VectorXd const &array2
    ) {
        std::vector<double> varray1(
            array1.data(), array1.data() + array1.size()
        );
        std::vector<double> varray2(
            array2.data(), array2.data() + array2.size()
        );
        return myMath::vector_dot(varray1, varray2);
    }
    
    long long transformation_id(
        MyTransformParams const &_params
    ) 
    {
        return logp_func.transformation_id(_params);
    }
};
