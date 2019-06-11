#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double delta = j[1]["steering_angle"];
          double a = j[1]["throttle"];

          /** calculate steering angle and throttle value
                      using MPC (both are in between [-1, 1]) **/

          /* transform the points in map coordinate
                  to the vehicle coordinate and save them */
          Eigen::VectorXd ptsx_car(ptsx.size()); // x coordinates of points in vehicle frame
          Eigen::VectorXd ptsy_car(ptsy.size()); // y coordinates of points in vehicle frame
          for(size_t i=0; i< ptsx.size(); i++){
            double dev_x = ptsx[i] - px;
            double dev_y = ptsy[i] - py;
            ptsx_car[i] = dev_x * cos(psi) + dev_y * sin(psi);
            ptsy_car[i] = (-1) * dev_x * sin(psi) + dev_y * cos(psi);
          }

          /* fitting the polynomial */
          const int order_poly = 3; // order of polynomial for fiting
          Eigen::VectorXd coeffs = polyfit(ptsx_car, ptsy_car, order_poly);
          // note that in vehicle frame the vehicle is at (x, y)=(0,0) and psi=0
          // the cross track error (for fitted function f(x), f(x=0) = coeffs[0])
          double cte = coeffs[0] - 0.0;
          // the orientation error (for fitted function f(x), f'(x=0) = coeffs[1])
          double epsi = 0.0 - atan(coeffs[1]);

          /* take into account latency to evaluate the state */
          Eigen::VectorXd state(6);
          // acurator delay time
          double t_delay = 100.0/1000.0; // time elapse due to latency (100ms = 0.1s)
          const double Lf = 2.67; // distance between center of mass of vehicle and the front axle
          // note that in vehicle frame the vehicle moves in x-direction (psi is 0 before adding latency)
          double px_delay = v * t_delay;
          double py_delay = 0.0;
          double psi_delay = - v * t_delay / Lf * delta;
          double v_delay = v + a * t_delay;
          double cte_delay = cte + v * t_delay * sin(epsi);
          double epsi_delay = epsi  -  v * t_delay / Lf * delta;

          state << px_delay, py_delay, psi_delay, v_delay, cte_delay, epsi_delay;

          /* evaluate the steering angle and throttle value */
          // create a state vector (with latency taken into account )
          auto vars = mpc.Solve(state, coeffs);
          double steer_value = vars[0] / deg2rad(25);
          double throttle_value = vars[1];

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25)] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          /* for displaying the MPC predicted trajectory (x, y).
              points are in reference to the vehicle's coordinate system.
              the points in the simulator are connected by a Green line. */
          vector<double> mpc_x_vals = {state[0]};
          vector<double> mpc_y_vals = {state[1]};

          for(size_t i=2; i<vars.size(); i+=2){
            mpc_x_vals.push_back(vars[i]);
            mpc_y_vals.push_back(vars[i+1]);
          }
          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          /* for displaying the waypoints/reference line.
              points are in reference to the vehicle's coordinate system.
              the points in the simulator are connected by Yellow line. */
          vector<double> next_x_vals;
          vector<double> next_y_vals;
          int num_points_ref = 25; // number of points
          double x_interval_ref = 2.5; // interval between two nearby points in x-direction

          for(int i=1; i< num_points_ref+1; i++){
            double next_x = i * x_interval_ref;
            next_x_vals.push_back(next_x);
            next_y_vals.push_back(polyeval(coeffs, next_x));
          }
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
