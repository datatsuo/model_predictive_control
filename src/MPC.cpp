#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

size_t N = 20; // the number of time steps for MPC
double dt = 1.0/N; //time elapse per time step (1.0/20 seconds = 50 milliseconds)

/*
This value assumes the model presented in the classroom is used.

It was obtained by measuring the radius formed by running the vehicle in the
simulator around in a circle with a constant steering angle and velocity on a
flat terrain.

Lf was tuned until the the radius formed by the simulating the model
presented in the classroom matched the previous radius.

This is the length from front to CoG that has a similar radius.
*/
const double Lf = 2.67; // distance between the center of mass of the vehicle and the front axle

const double ref_v = 80.0; // reference velocity
// indices to be used later
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // `fg` a vector of the cost  constraints,
    // `vars` is a vector of variable values (state & actuators)

    // weights for the terms in the cost
    const double cte2_weight = 20.0; // weight for cte^2
    const double epsi2_weight = 100.0; // weight for epsi^2
    const double v2_weight = 0.02; // weight for (velocity - reference velocity)^2
    const double delta2_weight = 1000.0; // weight for (steering angle)^2
    const double a2_weight = 1.0; // weight for (acceleration)^2
    const double dev_delta2_weight = 50000.0; //weight for (time diff of steering angle)^2
    const double dev_a2_weight = 5.0; //weight for (time diff of acceleration)^2

    /* compute the cost */
    // initialization
    fg[0] = 0.0;

    // cte^2, epsi^2, (velocity - reference velocity)^2 must be small
    for(size_t t=0; t<N; t++){
      fg[0] += cte2_weight * CppAD::pow(vars[cte_start + t], 2);
      fg[0] += epsi2_weight * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += v2_weight * CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // the actuator^2 must be small
    for(size_t t=0; t<N-1; t++){
      fg[0] += delta2_weight * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += a2_weight * CppAD::pow(vars[a_start + t], 2);
    }

    // the actuator^2 must not change drastically during a short time
    for(size_t t=0; t<N-2; t++){
      fg[0] += dev_delta2_weight * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += dev_a2_weight * CppAD::pow(vars[a_start + t+ 1] - vars[a_start + t], 2);
    }

    /* set initial constraints */
    fg[x_start + 1] = vars[x_start];
    fg[y_start + 1] = vars[y_start];
    fg[psi_start + 1] = vars[psi_start];
    fg[v_start + 1] = vars[v_start];
    fg[cte_start + 1] = vars[cte_start];
    fg[epsi_start + 1] = vars[epsi_start];

    /* set the rest of the constraints */
    for (size_t t = 1; t < N; t++) {

      // define some variables for writing down the constraints
      // at the current time step
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // at the previous time step
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // fitting polynomial evaluated at x0
      AD<double> f0 = 0.0;
      for(int i=0; i < coeffs.size(); i++){
        f0 += CppAD::pow(x0, i) * coeffs[i];
      }

      // psides evaluated at x0
      AD<double> psides0;
      AD<double> deriv_f0 = 0.0;
      for(int i=1; i < coeffs.size(); i++){
        deriv_f0 += i * coeffs[i] * CppAD::pow(x0, i-1);
      }
      psides0 = CppAD::atan(deriv_f0);

      // set the rest of the model constraints
      fg[x_start + t + 1] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[y_start + t + 1] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[psi_start + t + 1] = psi1 - (psi0 - v0 * dt / Lf * delta0);
      fg[v_start + t + 1] = v1 - (v0 + a0 * dt);
      fg[cte_start + t + 1] = cte1 - ((f0 - y0) + v0 * dt * CppAD::sin(epsi0));
      fg[epsi_start + t + 1] = epsi1 - ((psi0 - psides0) - v0 * dt / Lf * delta0);

    }

  }

};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {

  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // the number of model variables (both states and inputs included).
  size_t n_vars = 6 * N  + 2 * (N - 1);
  // the number of constraints
  size_t n_constraints =  6 * N;

  /* Initial value of the independent variables */
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (size_t i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }

  // store initial state to new variables for later use
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  /* set lower and upper limits for variables */
  Dvector vars_lowerbound(n_vars); // lower bounds
  Dvector vars_upperbound(n_vars); // upper bounds

  // upper and lower limits of non-actuators: max negative and positive values
  for (size_t i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // upper and lower limits of delta: -25 and 25 degrees (values in radians)
  for (size_t i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // upper and lower limits of acceleration/decceleration: 1.0 and -1.0
  for (size_t i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  /* set lower and upper limits for the constraints */
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (size_t i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0.0;
    constraints_upperbound[i] = 0.0;
  }

  // set the upper/lower limit for the constraints for the initial state
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  // segment error here
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  /* return the first actuator values,
    as well as the predicted values of x, y (to be used in main.cpp) */
  vector<double> result;

  // add the first accurator values
  result.push_back(solution.x[delta_start]);
  result.push_back(solution.x[a_start]);

  // add the predicted values of x, y
  for(size_t i=0; i < N; i++){
    result.push_back(solution.x[x_start + i]);
    result.push_back(solution.x[y_start + i]);
  }

  return result;

}
