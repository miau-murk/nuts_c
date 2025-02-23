#include <iostream>
#include <vector>
#include <cmath>
#include <random>
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
        std::vector< std::vector<double> > const &_untransformed_positions,
        std::vector< std::vector<double> > const &_untransformed_gradients,
        TransformParams &_params
    );
    
    TransformParams new_transformation(
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
        
    Eigen::MatrixXd new_eig_vectors(
        std::vector<std::vector<double> > vals
    ) 
    {
        int ndim = this->dim();
        int nvecs = (int) vals.size();
        Eigen::MatrixXd res(nvecs, ndim);
        for (int i = 0; i < nvecs; i++) {
            res.row(i) = Eigen::VectorXd::Map(
                vals[i].data(), 
                static_cast<Eigen::Index>(vals[i].size())
            );
        }
        return res.transpose();
    }
    
    Eigen::MatrixXd new_eig_values(std::vector<double> vals) 
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
        return mathCore::scalar_prods2(
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
        return mathCore::scalar_prods3(
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
        mathCore::axpy_out(vx, vy, a, vout);
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
        mathCore::axpy(vx, vy, a);
        y = Eigen::VectorXd::Map(
            vy.data(), static_cast<Eigen::Index>(vy.size())
        );
    }
    
    void fill_array(Eigen::VectorXd &array, double val) {
        array.fill(val);
    }
    
    bool array_all_finite(Eigen::VectorXd const &array) {
        bool res = true;
        for (double x: array) res &= std::isfinite(x);
        return res;
    }
    
    bool array_all_finite_and_nonzero(
        Eigen::VectorXd const &array
    ) 
    {
        bool res = true;
        for (double x: array) 
            res &= (std::isfinite(x) && x != 0);
        return res;
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
        mathCore::multiply(varray1, varray2, vdest);
        dest = Eigen::VectorXd::Map(
            vdest.data(), 
            static_cast<Eigen::Index>(vdest.size())
        );
    }
    
    void array_mult_eigs(
        Eigen::VectorXd const &stds,
        Eigen::VectorXd const &rhs,
        Eigen::VectorXd &dest,
        Eigen::MatrixXd const &vecs,
        Eigen::VectorXd const &vals
    )
    {
        Eigen::VectorXd rhs_ = stds.asDiagonal() * rhs;
        Eigen::VectorXd trafo = vecs.transpose() * rhs_;
        Eigen::VectorXd inner_prod = 
            vecs * (vals.asDiagonal() * trafo - trafo) + rhs_;
        dest = stds.asDiagonal() * inner_prod;
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
        return mathCore::vector_dot(varray1, varray2);
    }
    
    void array_gaussian(
        Eigen::VectorXd &dest,
        Eigen::VectorXd const &stds
    )
    {
        std::mt19937 gen(1701); 
        // TODO: сделать генерацию настройкой?
        
        std::normal_distribution<> dist{0., 1.};
        std::vector<double> vstds(
            stds.data(), stds.data() + stds.size()
        );
        std::vector<double> vdest(
            dest.data(), dest.data() + dest.size()
        );
        for (int i = 0; i < (int) vdest.size(); i++) {
            double norm = dist(gen);
            vdest[i] = vstds[i] * norm;
        }
        dest = Eigen::VectorXd::Map(
            vdest.data(), 
            static_cast<Eigen::Index>(vdest.size())
        );
    }
    
    void array_gaussian_eigs(
        Eigen::VectorXd &dest,
        Eigen::VectorXd const &scale,
        Eigen::MatrixXd const &vecs,
        Eigen::VectorXd const &vals
    )
    {
        std::mt19937 gen(1701); 
        // TODO: сделать генерацию настройкой?
        
        std::normal_distribution<> dist{0., 1.};
        std::vector<double> p(this->dim());
        for (int i = 0; i < (int) p.size(); i++)
            p[i] = dist(gen);
        Eigen::VectorXd draw = Eigen::VectorXd::Map(
            p.data(), 
            static_cast<Eigen::Index>(p.size())
        );
        
        Eigen::VectorXd trafo = vecs.transpose() * draw;
        Eigen::VectorXd inner_prod = 
            vecs * (vals.asDiagonal() * trafo - trafo) + draw;
        dest = scale.asDiagonal() * inner_prod;
    }
    
    void array_update_variance(
        Eigen::VectorXd &mean,
        Eigen::VectorXd &variance,
        Eigen::VectorXd const &value,
        double diff_scale // 1/self.count
    )
    {
        std::vector<double> mean_v(
            mean.data(), mean.data() + mean.size()
        );
        std::vector<double> variance_v(
            variance.data(), variance.data() + variance.size()
        );
        std::vector<double> value_v(
            value_v.data(), value_v.data() + value_v.size()
        );
        for (int i = 0; i < (int) value_v.size(); i++)
        {
            double diff = value_v[i] - mean_v[i];
            mean_v[i] += diff * diff_scale;
            variance_v[i] += diff * diff;
        }
        mean = Eigen::VectorXd::Map(
            mean_v.data(), 
            static_cast<Eigen::Index>(mean_v.size())
        );
        variance = Eigen::VectorXd::Map(
            variance_v.data(), 
            static_cast<Eigen::Index>(variance_v.size())
        );
    }
    
    void array_update_var_inv_std_draw(
        Eigen::VectorXd &variance_out,
        Eigen::VectorXd &inv_std,
        Eigen::VectorXd const &draw_var,
        double scale,
        std::pair<bool,double> fill_invalid,
        // в оригинале fill_invalid - nullable-тип, 
        // приходится что-то придумывать в альтернативу
        std::pair<double,double> clamp
    )
    {
        std::vector<double> variance_out_v(
            variance_out.data(), 
            variance_out.data() + variance_out.size()
        );
        std::vector<double> inv_std_v(
            inv_std.data(), inv_std.data() + inv_std.size()
        );
        std::vector<double> draw_var_v(
            draw_var.data(), draw_var.data() + draw_var.size()
        );
        for (int i = 0; i < (int) variance_out_v.size(); i++)
        {
            double draw_var_ = draw_var[i] * scale;
            if (!std::isfinite(draw_var_) || draw_var_ == 0)
            {
                if (fill_invalid.first) {
                    double fill_val = fill_invalid.second;
                    variance_out_v[i] = fill_val;
                    inv_std_v[i] = sqrt(1 / fill_val);
                }
                else {
                    double fill = draw_var_v[i];
                    fill = std::max(fill, clamp.first);
                    fill = std::min(fill, clamp.second);
                    variance_out_v[i] = fill;
                    inv_std_v[i] = sqrt(1 / fill);
                }
            }
        }
        variance_out = Eigen::VectorXd::Map(
            variance_out_v.data(), 
            static_cast<Eigen::Index>(variance_out_v.size())
        );
        inv_std = Eigen::VectorXd::Map(
            inv_std_v.data(), 
            static_cast<Eigen::Index>(inv_std_v.size())
        );
    }
    
    void array_update_var_inv_std_draw_grad(
        Eigen::VectorXd &variance_out,
        Eigen::VectorXd &inv_std,
        Eigen::VectorXd const &draw_var,
        Eigen::VectorXd const &grad_var,
        std::pair<bool,double> fill_invalid,
        // в оригинале fill_invalid - nullable-тип, 
        // приходится что-то придумывать в альтернативу
        std::pair<double,double> clamp
    )
    {
        std::vector<double> variance_out_v(
            variance_out.data(), 
            variance_out.data() + variance_out.size()
        );
        std::vector<double> inv_std_v(
            inv_std.data(), inv_std.data() + inv_std.size()
        );
        std::vector<double> draw_var_v(
            draw_var.data(), draw_var.data() + draw_var.size()
        );
        std::vector<double> grad_var_v(
            grad_var.data(), grad_var.data() + grad_var.size()
        );
        for (int i = 0; i < (int) variance_out_v.size(); i++)
        {
            double val = sqrt(draw_var[i] / grad_var[i]);
            if (!std::isfinite(val) || val == 0)
            {
                if (fill_invalid.first) {
                    double fill_val = fill_invalid.second;
                    variance_out_v[i] = fill_val;
                    inv_std_v[i] = sqrt(1 / fill_val);
                }
                else {
                    double fill = draw_var_v[i];
                    fill = std::max(fill, clamp.first);
                    fill = std::min(fill, clamp.second);
                    variance_out_v[i] = fill;
                    inv_std_v[i] = sqrt(1 / fill);
                }
            }
        }
        variance_out = Eigen::VectorXd::Map(
            variance_out_v.data(), 
            static_cast<Eigen::Index>(variance_out_v.size())
        );
        inv_std = Eigen::VectorXd::Map(
            inv_std_v.data(), 
            static_cast<Eigen::Index>(inv_std_v.size())
        );
    }
    
    void array_update_var_inv_std_grad(
        Eigen::VectorXd &variance_out,
        Eigen::VectorXd &inv_std,
        Eigen::VectorXd const &gradient,
        double fill_invalid,
        std::pair<double,double> clamp
    )
    {
        std::vector<double> variance_out_v(
            variance_out.data(), 
            variance_out.data() + variance_out.size()
        );
        std::vector<double> inv_std_v(
            inv_std.data(), inv_std.data() + inv_std.size()
        );
        std::vector<double> gradient_v(
            gradient.data(), gradient.data() + gradient.size()
        );
        for (int i = 0; i < (int) variance_out_v.size(); i++)
        {
            double val = abs(gradient_v[i]);
            val = std::max(val, clamp.first);
            val = std::min(val, clamp.second);
            val = 1 / val;
            if (!std::isfinite(val))
                val = fill_invalid;
            variance_out_v[i] = val;
            inv_std_v[i] = sqrt(1 / val);
        }
        variance_out = Eigen::VectorXd::Map(
            variance_out_v.data(), 
            static_cast<Eigen::Index>(variance_out_v.size())
        );
        inv_std = Eigen::VectorXd::Map(
            inv_std_v.data(), 
            static_cast<Eigen::Index>(inv_std_v.size())
        );
    }
    
    std::vector<double> eigs_as_array(
        Eigen::VectorXd const &source
    )
    {
        return std::vector<double>(
            source.data(), source.data() + source.size()
        );
    }
    
    double inv_transform_normalize(
        MyTransformParams const &params,
        Eigen::VectorXd const &untransformed_position,
        Eigen::VectorXd const &untransformed_gradient,
        Eigen::VectorXd &transformed_position,
        Eigen::VectorXd &transformed_gradient
    )
    {
        std::vector<double> untransformed_position_v(
            untransformed_position.data(), 
            untransformed_position.data() + untransformed_position.size()
        );
        std::vector<double> untransformed_gradient_v(
            untransformed_gradient.data(), 
            untransformed_gradient.data() + untransformed_gradient.size()
        );
        std::vector<double> transformed_position_v(
            transformed_position.data(), 
            transformed_position.data() + transformed_position.size()
        );
        std::vector<double> transformed_gradient_v(
            transformed_gradient.data(), 
            transformed_gradient.data() + transformed_gradient.size()
        );
        double res = logp_func.inv_transform_normalize(
            params,
            untransformed_position_v,
            untransformed_gradient_v,
            transformed_position_v,
            transformed_gradient_v
        );
        transformed_position = Eigen::VectorXd::Map(
            transformed_position_v.data(), 
            static_cast<Eigen::Index>(transformed_position_v.size())
        );
        transformed_gradient = Eigen::VectorXd::Map(
            transformed_gradient_v.data(), 
            static_cast<Eigen::Index>(transformed_gradient_v.size())
        );
        return res;
    }
    
    std::pair<double,double> init_from_untransformed_position(
        MyTransformParams const &params,
        Eigen::VectorXd const &untransformed_position,
        Eigen::VectorXd &untransformed_gradient,
        Eigen::VectorXd &transformed_position,
        Eigen::VectorXd &transformed_gradient
    )
    {
        std::vector<double> untransformed_position_v(
            untransformed_position.data(), 
            untransformed_position.data() + untransformed_position.size()
        );
        std::vector<double> untransformed_gradient_v(
            untransformed_gradient.data(), 
            untransformed_gradient.data() + untransformed_gradient.size()
        );
        std::vector<double> transformed_position_v(
            transformed_position.data(), 
            transformed_position.data() + transformed_position.size()
        );
        std::vector<double> transformed_gradient_v(
            transformed_gradient.data(), 
            transformed_gradient.data() + transformed_gradient.size()
        );
        std::pair<double,double> res = 
        logp_func.init_from_untransformed_position(
            params,
            untransformed_position_v,
            untransformed_gradient_v,
            transformed_position_v,
            transformed_gradient_v
        );
        untransformed_gradient = Eigen::VectorXd::Map(
            untransformed_gradient_v.data(), 
            static_cast<Eigen::Index>(untransformed_gradient_v.size())
        );
        transformed_position = Eigen::VectorXd::Map(
            transformed_position_v.data(), 
            static_cast<Eigen::Index>(transformed_position_v.size())
        );
        transformed_gradient = Eigen::VectorXd::Map(
            transformed_gradient_v.data(), 
            static_cast<Eigen::Index>(transformed_gradient_v.size())
        );
        return res;
    }
    
    std::pair<double,double> init_from_transformed_position(
        MyTransformParams const &params,
        Eigen::VectorXd &untransformed_position,
        Eigen::VectorXd &untransformed_gradient,
        Eigen::VectorXd const &transformed_position,
        Eigen::VectorXd &transformed_gradient
    )
    {
        std::vector<double> untransformed_position_v(
            untransformed_position.data(), 
            untransformed_position.data() + untransformed_position.size()
        );
        std::vector<double> untransformed_gradient_v(
            untransformed_gradient.data(), 
            untransformed_gradient.data() + untransformed_gradient.size()
        );
        std::vector<double> transformed_position_v(
            transformed_position.data(), 
            transformed_position.data() + transformed_position.size()
        );
        std::vector<double> transformed_gradient_v(
            transformed_gradient.data(), 
            transformed_gradient.data() + transformed_gradient.size()
        );
        std::pair<double,double> res = 
        logp_func.init_from_transformed_position(
            params,
            untransformed_position_v,
            untransformed_gradient_v,
            transformed_position_v,
            transformed_gradient_v
        );
        untransformed_position = Eigen::VectorXd::Map(
            transformed_position_v.data(), 
            static_cast<Eigen::Index>(transformed_position_v.size())
        );
        untransformed_gradient = Eigen::VectorXd::Map(
            untransformed_gradient_v.data(), 
            static_cast<Eigen::Index>(untransformed_gradient_v.size())
        );
        transformed_gradient = Eigen::VectorXd::Map(
            transformed_gradient_v.data(), 
            static_cast<Eigen::Index>(transformed_gradient_v.size())
        );
        return res;
    }
    
    void update_transformation(
        std::vector< Eigen::VectorXd > const &untransformed_positions,
        std::vector< Eigen::VectorXd > const &untransformed_gradients,
        MyTransformParams &params
    )
    {
        std::vector< std::vector<double> > untransformed_positions_v;
        std::vector< std::vector<double> > untransformed_gradients_v;
        
        for (int i = 0; i < (int)untransformed_positions.size(); i++)
        {
            untransformed_positions_v.push_back(
                std::vector<double>(
                    untransformed_positions[i].data(), 
                    untransformed_positions[i].data() + untransformed_positions[i].size()
                )
            );
            untransformed_gradients_v.push_back(
                std::vector<double>(
                    untransformed_gradients[i].data(), 
                    untransformed_gradients[i].data() + untransformed_gradients[i].size()
                )
            );
        }
        
        logp_func.update_transformation(
            untransformed_positions_v,
            untransformed_gradients_v,
            params
        );
    }
    
    MyTransformParams new_transformation(
        Eigen::VectorXd const &untransformed_position,
        Eigen::VectorXd const &untransformed_gradient,
        unsigned long long chain
    )
    {
        std::vector<double> untransformed_position_v(
            untransformed_position.data(), 
            untransformed_position.data() + untransformed_position.size()
        );
        std::vector<double> untransformed_gradient_v(
            untransformed_gradient.data(), 
            untransformed_gradient.data() + untransformed_gradient.size()
        );
        return logp_func.new_transformation(
            untransformed_position_v,
            untransformed_gradient_v,
            chain
        );
    }
    
    long long transformation_id(
        MyTransformParams const &_params
    ) 
    {
        return logp_func.transformation_id(_params);
    }
};
