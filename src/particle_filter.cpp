/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <random>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  //std::default_random_engine gen;
  std::random_device rand;
  std::mt19937 gen(rand()); 
  normal_distribution<double> dist_x(0, std[0]);
  normal_distribution<double> dist_y(0, std[1]);
  normal_distribution<double> dist_theta(0, std[2]);
  num_particles = 100;  // TODO: Set the number of particles

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x + dist_x(gen);
    p.y = y + dist_y(gen);
    p.theta = theta + dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < particles.size(); i++) {
    if (fabs(yaw_rate) > 0.0001) {
      Particle p = particles[i];
      particles[i].x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      particles[i].y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      particles[i].theta += delta_t * yaw_rate;
    } else {
      Particle p = particles[i];
      particles[i].x += velocity * delta_t * cos(p.theta);
      particles[i].y += velocity * delta_t * sin(p.theta);
    }
     particles[i].x += dist_x(gen);
     particles[i].y += dist_y(gen);
     particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  int idx = 0;
  for(int i = 0; i < observations.size(); i++) {
    double distance_smallest = std::numeric_limits<double>::max();
    int map_id = -1;
    for(int j = 0; j < predicted.size(); j++) {
      double distance = sqrt(pow((predicted[j].x - observations[i].x), 2) + pow((predicted[j].y - observations[i].y), 2));
      if (distance_smallest > distance) {
        distance_smallest = distance;
        map_id = predicted[j].id;
      }
    }
    observations[i].id = map_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    int p_id = particles[i].id;
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> predicted;
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double m_x = map_landmarks.landmark_list[j].x_f;
      double m_y = map_landmarks.landmark_list[j].y_f;
      int m_id = map_landmarks.landmark_list[j].id_i;
      if (fabs(p_x - m_x) <= sensor_range && fabs(p_y - m_y) <= sensor_range) {
        predicted.push_back(
          LandmarkObs{
            m_id,
            m_x,
            m_y
          }
        );
      }
    }
    
    vector<LandmarkObs> trans_observations;
    for (int j = 0; j < observations.size(); j++) {
      double o_x = observations[j].x;
      double o_y = observations[j].y;

      double t_x = cos(p_theta)*o_x - sin(p_theta)*o_y + p_x;
      double t_y = sin(p_theta)*o_x + cos(p_theta)*o_y + p_y;
      trans_observations.push_back(
        LandmarkObs{
          observations[j].id,
          t_x,
          t_y
        }
      );
    }
    
    dataAssociation(predicted, trans_observations);
    particles[i].weight = 1.0;

    for (int k = 0; k < trans_observations.size(); k++) {
      for (int j = 0; j < predicted.size(); j++) {
        if (trans_observations[k].id == predicted[j].id) {
          particles[i].weight *= 1/(fabs(2*M_PI*std_landmark[0]*std_landmark[1]))
            *exp(-(pow(predicted[j].x-trans_observations[k].x, 2)/(2*pow(std_landmark[0],2)) 
            + pow(predicted[j].y-trans_observations[k].y, 2)/(2*pow(std_landmark[1],2))));
      
        }
      }
            
    }
    weights.push_back(particles[i].weight);
  }
  


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distr(0, num_particles-1);
  double max_v = *max_element(weights.begin(), weights.end());

  std::uniform_real_distribution<double> distrR(0.0, max_v);

  double beta = 0.0;
  int idx = distr(gen);
  std::vector<Particle> tmp_particles;
  for (int i = 0; i < num_particles; i++) {
    beta += distrR(gen) * 2;
    while(beta > particles[idx].weight) {
      beta -= particles[idx].weight;
      idx = (idx + 1) % num_particles;
    }
    tmp_particles.push_back(particles[idx]);
  }
  particles = tmp_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
