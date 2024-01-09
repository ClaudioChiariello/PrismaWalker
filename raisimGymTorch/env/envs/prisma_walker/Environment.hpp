//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include "Eigen/Eigen"
#include "motors_src/lookup.hpp"
#include "motors_src/trajectory.hpp"


#include "motors_src/group_command.hpp"
#include "motors_src/group_feedback.hpp"
#include "motors_src/command.hpp"

#include <cstdlib>
#include "raisim/World.hpp"
#include <fstream>
#include <vector> 
#include "lookup.hpp"
#include "../../RaisimGymEnv.hpp"
#include "raisim/contact/Contact.hpp"
#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <termios.h>
#define STDIN_FILENO 0
#elif defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#endif
#include <stdio.h>
#include "dynamixel_sdk.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
// Control table address
#define ADDR_PRO_TORQUE_ENABLE          64                 // Control table address is different in Dynamixel model
#define ADDR_PRO_GOAL_POSITION          116
#define ADDR_PRO_PRESENT_POSITION       132
#define ADDR_PRO_PRESENT_VELOCITY       128

// Protocol version
#define PROTOCOL_VERSION                2.0                 // See which protocol version is used in the Dynamixel

// Default setting
#define DXL_ID                          1                   // Dynamixel ID: 1
#define BAUDRATE                        2000000
#define DEVICENAME                      "/dev/ttyUSB0"      // Check which port is being used on your controller
															// ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

#define TORQUE_ENABLE                   1                   // Value for enabling the torque
#define TORQUE_DISABLE                  0                   // Value for disabling the torque
#define DXL_MINIMUM_POSITION_VALUE      0            // Dynamixel will rotate between this value
#define DXL_MAXIMUM_POSITION_VALUE      4095             // and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
#define DXL_MOVING_STATUS_THRESHOLD     20                  // Dynamixel moving status threshold

#define ESC_ASCII_VALUE                 0x1b
namespace raisim {
 #define N 20
 
	
//#define num_row_in_file 4000
//#define decimal_precision 8
class ENVIRONMENT : public RaisimGymEnv {

 public:

	explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
		RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

		/// create world
		world_ = std::make_unique<raisim::World>();

		/// add objects
		prisma_walker = world_->addArticulatedSystem("/home/claudio/raisim_ws/raisimlib/rsc/prisma_walker/urdf/prisma_walker.urdf");
		prisma_walker->setName("prisma_walker");
		prisma_walker->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
		world_->addGround();
		Eigen::Vector3d gravity_d(0.0,0.0,0);
		/// get robot data
		gcDim_ = prisma_walker->getGeneralizedCoordinateDim();
		gvDim_ = prisma_walker->getDOF();
		nJoints_ = gvDim_ - 6;

		/// initialize containers
		gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
		gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
		pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget3_.setZero(nJoints_);  
		roll_init_= 0.0;pitch_init_=0.0;yaw_init_=0.0;
		q_ = Eigen::AngleAxisf(roll_init_, Eigen::Vector3f::UnitX())
		* Eigen::AngleAxisf(pitch_init_, Eigen::Vector3f::UnitY())
		* Eigen::AngleAxisf(yaw_init_, Eigen::Vector3f::UnitZ());
		/// this is nominal configuration of prisma walker
		gc_init_ << 0.0, 0.0, 0.33, q_.w(), q_.x(), q_.y(), q_.z(), 0.6,0.6,0.0; //(pos, orientament, configurazione)

		/// set pd gains
		Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
		jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(20.0);
		jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(2.0);
		//usando un d gain 0.2 non converge a niente, mettendolo a 2 invece per un po' sembra migliorare ma poi torna a peggiorare

		READ_YAML(int, num_seq, cfg_["num_seq"]);
   		READ_YAML(int, num_seq_vel, cfg_["num_seq_vel"]);
 
		joint_history_pos_.setZero(3*num_seq);
    	joint_history_vel_.setZero(3*num_seq_vel);
		current_action_.setZero(3);

		prisma_walker->setPdGains(jointPgain, jointDgain);
		prisma_walker->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

		/// MUST BE DONE FOR ALL ENVIRONMENTS
		obDim_ = 16 + joint_history_pos_.size() + joint_history_vel_.size() + current_action_.size();
		actionDim_ = nJoints_; 
		
		actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
		obDouble_.setZero(obDim_);

		/// action scaling
		actionMean_ = gc_init_.tail(nJoints_);
		double action_std;

		command_<< 0.2, 0, 0.12;
		READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
		
		// Open port
		actionStd_.setConstant(action_std);
		NumberIteration_ = 0;
		/// Reward coefficients
		rewards_.initializeFromConfigurationFile (cfg["reward"]);

		footIndex_ = prisma_walker->getBodyIdx("link_3");
		
		foot_center_ = prisma_walker->getFrameIdxByLinkName("piede_interno"); //since there is a fixed joint, I cant assign a body idx to it
		foot_sx_ = prisma_walker->getFrameIdxByLinkName("piede_sx");
		foot_dx_ = prisma_walker->getFrameIdxByLinkName("piede_dx");

		m1_pos_(1818);
		m2_pos_(1818); 

		/// visualize if it is the first environment
		if (visualizable_) {
		server_ = std::make_unique<raisim::RaisimServer>(world_.get());
		server_->launchServer();
		server_->focusOn(prisma_walker);
		}

		
	}


	void command_vel(double v_x, double v_y, double omega_z){ //This function can be called to declare a new command velocity
		command_[0] = v_x;
		command_[1] = v_y;
		command_[2] = omega_z;
		std::cout<<"command_vel: "<<command_<<std::endl;
		
	}

	void init() final { 
	}     

	void reset() final {
		
		previous_height_ = 0;
		alfa_z_ = 0;
		max_height_ = 0;
		clearance_foot_ = 0.1; //penalita' iniziale. Altrimenti non alzera' mai il piede

		std::fstream m1_traj;
    	std::fstream m2_traj;
		int num_row_in_file = 1818;
		m1_traj.open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m1_18s.txt",std::ios::in);
    	m2_traj.open("/home/claudio/raisim_ws/raisimlib/raisimGymTorch/raisimGymTorch/env/envs/prisma_walker/pos_m2_18s.txt",std::ios::in);
	
		Eigen::VectorXd m1_pos(num_row_in_file);
		Eigen::VectorXd m2_pos(num_row_in_file);


		if(m1_traj.is_open() && m2_traj.is_open()){
			for(int j = 0;j<num_row_in_file;j++)
				m1_traj >> m1_pos(j);  //one character at time store the content of the file inside the vector

			m1_pos_=-m1_pos;

			for(int j = 0;j<num_row_in_file;j++)
				m2_traj >> m2_pos(j);

			m2_pos_=-m2_pos;
		}
		m1_traj.close(); 
		index_imitation_ = 0;
		previous_contact = 0;
		prisma_walker->setState(gc_init_, gv_init_);
		updateObservation();
		NumberIteration_ = 0;
		vel_rew_ = 0;
		mean_vel_rew_ = 10;
		//swing_time_d = -10; Da usare solo in caso di penalty
		previous_contact = 1; //Parte da piede a terra e quindi va a 1
	}
	

 
	float step(const Eigen::Ref<EigenVec>& action) final {
		/// action scaling
		pTarget3_ = action.cast<double>(); //dim=n_joints
		pTarget3_ = pTarget3_.cwiseProduct(actionStd_);
		actionMean_ << m1_pos_(index_imitation_), m2_pos_(index_imitation_), 0.0;
		pTarget3_ += actionMean_;
		pTarget_.tail(nJoints_) << pTarget3_;
		current_action_ = pTarget3_;
		
		//if ( gc_[7] - azione_precedente_[0] > azione_precedente_[0] -   ) //se la distanza tra dove sono e dove dovevo andare prima
		/*pTarget3_ << azione_precedente_[0]-((azione_precedente_[0]+pTarget3_[0])*smoothing_factor_),
					azione_precedente_[1]-((azione_precedente_[1]+pTarget3_[1])*smoothing_factor_),
					azione_precedente_[2]-((azione_precedente_[2]-pTarget3_[2])*smoothing_factor_);*/

		azione_precedente_ << pTarget3_[0], pTarget3_[1] ,pTarget3_[2];

       
		prisma_walker->setPdTarget(pTarget_, vTarget_);


		for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
			if(server_) server_->lockVisualizationServerMutex();
			world_->integrate();
			if(server_) server_->unlockVisualizationServerMutex();
		}

		updateObservation(); //Ricordati che va prima di fare tutti i conti
		prisma_walker->getFrameVelocity(foot_center_, vel_);
		prisma_walker->getFramePosition(foot_center_, footPosition_);
		prisma_walker->getFramePosition(foot_sx_, footPosition_Sx_);
		prisma_walker->getFramePosition(foot_dx_, footPosition_Dx_);
		
		contacts(); //va dopo il world integrate, perche' il world integrate aggiorna i contatti. 
		error_vel();

		rewards_.record("torque", prisma_walker->getGeneralizedForce().squaredNorm());
		rewards_.record("lin_vel", lin_reward_);
		rewards_.record("ang_vel", ang_reward_);

		//rewards_.record("lin_vel", mean_vel_rew_);
		rewards_.record("third_joint", abs(gc_[9]));
		rewards_.record("imitation", imitation_function(gc_[7], gc_[8]));
		rewards_.record("angular_penalty", 0.05*(bodyAngularVel_[0] + bodyLinearVel_[1])); //+0.025*bodyLinearVel_[2]
		rewards_.record("slipping_piede_interno", slip_term_ + ang_vel_term_contact_);
		rewards_.record("slipping_external_feet", slipping_external_feet_function());
		rewards_.record("clearance", clearance_foot_);
		//rewards_.record("air_foot", swing_time_.count()/1000);
		if(alfa_z_ > 65)
			rewards_.record("leaning", alfa_z_*alfa_z_);

		return rewards_.sum();
		
	}


	float imitation_function (double m1, double m2){
		index_imitation_ = index_imitation_+1;
		if(index_imitation_ == 1818)
			index_imitation_ = 0;
		return std::exp(-5* (std::pow(m1-m1_pos_(index_imitation_),2) + pow( m2- m2_pos_(index_imitation_) , 2)) );

	}
 
	float norm(Eigen::Vector3d vector){
		float norma=0;
		norma = sqrt(pow(vector(0),2)+pow(vector(1),2)+pow(vector(2),2));

		return norma;
	}


	void clearance(){
		if(cF_["center_foot"]==0){  //Swing phase
			if(previous_height_ > footPosition_[2]){// descent phase
				if(max_clearence_){ //during all the descendant phase, the agent receive the same penalty because wasnt able to lift enough the foot
					max_height_ = previous_height_;
					max_clearence_ = false;
					//RSINFO_IF(visualizable_, "clearance: ", clearance_foot_)
				}
			}else{  //rising phase 
				previous_height_ = footPosition_[2];    
				max_clearence_ = true;
			}	
		}
		else
			previous_height_ = 0;


		clearance_foot_ = (max_height_-0.12)*(max_height_-0.12);  //x<<y x*pow(2,y)

	}

	float slipping_external_feet_function(){ //when the fyll body lands, we penalize velocities
		Eigen::Vector3d v_xyz;
		Eigen::Vector3d w_xyz;
		v_xyz << gv_[0],gv_[1],gv_[2];
		w_xyz << gv_[3],gv_[4],gv_[5];
		if(cF_["lateral_feet"]==1)
			return norm(v_xyz)+norm(w_xyz);
		else 
			return 0;
	}

	void slippage(){
		
		prisma_walker->getFrameAngularVelocity(foot_center_, ang_vel_);
   		
		/*slip_term_sxdx_ = 0;
		if(cF_["lateral_feet"]==1 || (footPosition_Dx_[2]<0.001 && footPosition_Sx_[2]<0.001)){
			double lin_vel_sx_ = std::sqrt(vel_sx_[0]*vel_sx_[0] + vel_sx_[1]*vel_sx_[1]);
			double lin_vel_dx_ = std::sqrt(vel_dx_[0]*vel_dx_[0] + vel_dx_[1]*vel_dx_[1]);
			slip_term_sxdx_ += lin_vel_sx_ + lin_vel_dx_;
		}*/

		slip_term_ = 0;
		ang_vel_term_contact_ = 0;
		if(cF_["center_foot"] == 1 || footPosition_[2] < 0.001){
			slip_term_ += std::sqrt(vel_[0]*vel_[0] + vel_[1]*vel_[1]);
			ang_vel_term_contact_ = ang_vel_.e().squaredNorm();
		}
	}


	void FrictionCone(){
		int k = 4;
		int lambda = 4; //less than 1 gives some errors
		Eigen::Vector3d contactForce_L, contactForce_R, contactForce;
		double pi_j_k;
		double theta = std::atan(0.9);
		double alpha_L, alpha_R, alpha;

		cF_["center_foot"] = 0;
		cF_["lateral_feet"] = 0;

		H_ = 0;
		for(auto& contact: prisma_walker->getContacts()){
			if (contact.skip()) continue;  //contact.skip() ritorna true se siamo in una self-collision, in quel caso ti ritorna il contatto 2 volte
			if(contact.getlocalBodyIndex() == 3 ){
				cF_["center_foot"] = 1; 

				contactForce = (contact.getContactFrame().e().transpose() * contact.getImpulse().e()) / world_->getTimeStep();
				contactForce.normalize(); //a quanto pare e' stata implementata come void function, quindi non ha un return. D'ora in poi contactForce avra' norma 1
				alpha = std::acos( contact.getNormal().e().dot(contactForce));

				H_ = 1/(lambda*( (theta-alpha) * (theta+alpha)));
			}

			else if(contact.getlocalBodyIndex() == 0 ){  
				cF_["lateral_feet"] = 1; 
			}      
		} 

		slippage();
		clearance();
    }



	void error_vel(){
		if(cF_["center_foot"]==1){
			double err_x = command_[0] - bodyLinearVel_[0];
			double err_omega = command_[2] - bodyAngularVel_[2];
			lin_reward_ =  std::exp(-5*err_x*err_x);
			ang_reward_ =  std::exp(-15*err_omega*err_omega);
			previous_reward_ = lin_reward_ + ang_reward_;
		}else{		
			if(swing_time_ >= std::chrono::duration<double, std::milli>(3500) ){
				lin_reward_ = -0.1;
				ang_reward_ = -0.1;
			}
		}

	}

	void mean_square_error(){
		double err_x = command_[0] - bodyLinearVel_[0];
		double err_omega = command_[2] - bodyAngularVel_[2];
		vel_rew_ += (err_x*err_x) + (err_omega*err_omega);   //se la fase di swing dura poco fa poche iterazioni!!!!!!
	}

	void contacts(){
		
		cF_["center_foot"] = 0;
		cF_["lateral_feet"] = 0;
		for(auto& contact: prisma_walker->getContacts()){
			if (contact.skip()) continue;  //contact.skip() ritorna true se siamo in una self-collision, in quel caso ti ritorna il contatto 2 volte
			if(contact.getlocalBodyIndex() == 3 ){
				cF_["center_foot"] = 1;
			} 
			if(contact.getlocalBodyIndex() == 0 ){  
				cF_["lateral_feet"] = 1; 
			}      
		}

		mse_and_airTime();
	} 

	inline void mse_and_airTime(){

		if(cF_["center_foot"] == 1){  //NON TENERE LE COSE DENTRO IL FOR, PERCHè Altrimenti chiama le stesse funzioni piu' VOLTE!!!
			if(previous_contact == 0 && cF_["center_foot"]==1){ //piede atterra
				swing_time_ = std::chrono::duration<double, std::milli>(0);
			}
			lift_instant_ = std::chrono::steady_clock::now();
			//mean_square_error();
			NumberIteration_ ++;
		}

		previous_reward_ = 0;
	
		if(cF_["center_foot"] == 0){
			land_instant_ = std::chrono::steady_clock::now();
			swing_time_ += std::chrono::duration<double, std::milli>(land_instant_ - lift_instant_);
			lift_instant_ = std::chrono::steady_clock::now();

			/*if(previous_contact == 1 && cF_["center_foot"]==0){// Inizia a sollevare-> Robot Fermo
				bool start = true;
				
				/*mean_vel_rew_ = rew_x_ / NumberIteration_; //Riempe previous vel con l'errore medio
				NumberIteration_ = 1;
				vel_rew_ = 0;
			}		*/
		}

		previous_contact = cF_["center_foot"];		

		//RSINFO_IF(visualizable_, "swing time: "<< swing_time_.count())
		//RSINFO_IF(visualizable_, "error vel: "<< mean_rew_x_)

		slippage();
		clearance();

	}



	void generate_command_velocity(){ 
		std::random_device rd;  // Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<> dis(0.1, 0.4);

		command_[0] = dis(gen);
		command_[2] = 0.5*dis(gen);

		/*if(visualizable_){
			std::cout<<"Command velocity: "<<command_<<std::endl;
		}*/
	}
		

	void updateObservation() {
		
		prisma_walker->getState(gc_, gv_);//generalized coordinate generalized velocity wrt initial coordinate gc_init
		updateJointHistory();

		quat_[0] = gc_[3]; 
		quat_[1] = gc_[4]; 
		quat_[2] = gc_[5]; 
		quat_[3] = gc_[6];
		raisim::quatToRotMat(quat_, rot_);
		bodyLinearVel_ = rot_.e().transpose() * gv_.segment(0, 3);
		bodyAngularVel_ = rot_.e().transpose() * gv_.segment(3, 3);
	
		obDouble_ << gc_[2],
        rot_.e().row(1).transpose(),
		rot_.e().row(2).transpose(), /// body orientation e' chiaro che per la camminata, e' rilevante sapere come sono orientato rispetto all'azze z, e non a tutti e 3. L'orientamento rispetto a x e' quanto sono chinato in avanti, ma io quando cammino scelgo dove andare rispetto a quanto sono chinato??? ASSOLUTAMENTE NO! Anzi, cerco di non essere chinato. Figurati un orentamento rispetto a y, significherebbe fare la ruota sul posto
		bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
		command_,
		joint_history_pos_, /// joint angles
        joint_history_vel_,
		current_action_; 
	}


	void updateJointHistory(){

		Eigen::VectorXd temp_pos (3*num_seq);
		temp_pos << joint_history_pos_; //temp conterrà nelle posizioni 0-11 quello che joints_history conterra' nelle posizioni 12-23
		
		Eigen::VectorXd temp_vel (3*num_seq_vel);
		temp_vel << joint_history_vel_;

		for(int i = 0; i < (num_seq-1); i++){
		joint_history_pos_(Eigen::seq((i)*3, (i+1)*3-1)) = temp_pos(Eigen::seq((i+1)*3, (i+2)*3-1)); //overwrite the next sequence
		}

		for(int i = 0; i < (num_seq_vel-1); i++){
		joint_history_vel_(Eigen::seq((i)*3, (i+1)*3-1)) = temp_vel(Eigen::seq((i+1)*3, (i+2)*3-1));
		}

		//PER FARE DELLE PROVE, USA QUESTI VETTORI CHE HANNO SOLO NUMERI DA 1 a 11
			/*joint_history_pos_.tail(3) = Eigen::ArrayXd::LinSpaced(3, 0, 2); //genera un vettore di 12 elementi di numeri equidistanziati che vanno da 0 a 12
			joint_history_vel_.tail(3) = Eigen::ArrayXd::LinSpaced(3, 0, 2);
			if(visualizable_){
				std::cout<<"Joint pos: "<< joint_history_pos_<< std::endl;
				std::cout<<"Joint vel: "<< joint_history_vel_<< std::endl;
			}*/
		joint_history_pos_.tail(3) = gc_.tail(3);
		joint_history_vel_.tail(3) = gv_.tail(3);

		//Reshaping della time series
		/*for(int i = 0; i< 3; i++){
		for(int j = 0; j<num_seq; j++){
			joint_history_pos_reshaped_(Eigen::seq(i*num_seq, (i+1)*(num_seq-1)))[j] = joint_history_pos_(Eigen::seq(j*3, (j+1)*3))[i];
		}
		}

		for(int i = 0; i< 3; i++){
		for(int j = 0; j<num_seq_vel; j++){
			joint_history_vel_reshaped_(Eigen::seq(i*num_seq_vel, (i+1)*(num_seq_vel-1)))[j] = joint_history_vel_(Eigen::seq(j*3, (j+1)*3))[i];
		}
		}*/
	}


	void observe(Eigen::Ref<EigenVec> ob) final {
		/// convert it to float
		ob = obDouble_.cast<float>();
		//std::cout << "ob double: " << ob.size() << std::endl;
	}

	bool isTerminalState(float& terminalReward) final {
		terminalReward = float(terminalRewardCoeff_);
		
		Eigen::Vector3d z_vec=Eigen::Vector3d::Zero();
		if(tester_real){
			z_vec = R_imu_base_frame_.col(0);
		}else{
			z_vec = rot_.e().row(2).transpose();
		}
		Eigen::Vector3d z_axis(0,0,1);
		alfa_z_ = acos((z_vec.dot(z_axis))/norm(z_vec));
		alfa_z_ = (alfa_z_*180)/M_PI;
		std::chrono::steady_clock::time_point end_alfa_time = std::chrono::steady_clock::now();
		if(alfa_z_ > 50){
			elapsed_time_[2] += std::chrono::duration<double, std::milli>(end_alfa_time - begin_[2]); 
			begin_[2] =  std::chrono::steady_clock::now();
		}
		else{
			elapsed_time_[2] = std::chrono::duration<double, std::milli>(0);
			begin_[2] =  std::chrono::steady_clock::now();
		}
		//RSINFO_IF(visualizable_, alfa_z_*alfa_z_)
		if(alfa_z_>90)
			return true;
			
		std::chrono::steady_clock::time_point end[2] = {std::chrono::steady_clock::now(),
														std::chrono::steady_clock::now()};
														
		
		if(footPosition_[2] < 0.001){ 
			elapsed_time_[1] += std::chrono::duration<double, std::milli>(end[1] - begin_[1]); 
			begin_[1] =  std::chrono::steady_clock::now();
		}
		else{
			elapsed_time_[1] = std::chrono::duration<double, std::milli>(0);
			begin_[1] =  std::chrono::steady_clock::now();
		}

		if(footPosition_Dx_[2]<0.001 && footPosition_Sx_[2]<0.001){
			elapsed_time_[0] += std::chrono::duration<double, std::milli>(end[0] - begin_[0]); 
			begin_[0] =  std::chrono::steady_clock::now();
		}
		else{
			elapsed_time_[0] = std::chrono::duration<double, std::milli>(0);
			begin_[0] =  std::chrono::steady_clock::now();
		}
	
		//RSINFO_IF(visualizable_, elapsed_time_[0].count())
		if(elapsed_time_[1] >= std::chrono::duration<double, std::milli>(7500) 
			|| elapsed_time_[0] >= std::chrono::duration<double, std::milli>(7500)
			|| elapsed_time_[2] >= std::chrono::duration<double, std::milli>(3000)){ //3 secondi
			for(int i = 0; i<3; i++)
				elapsed_time_[i] = std::chrono::duration<double, std::milli>(0);

			RSINFO_IF(visualizable_, "locked")
			return true;
		}
		
	
		terminalReward = 0.f;
		return false;
	}

	void curriculumUpdate() {
		generate_command_velocity(); //La metto qui perche' la reset viene chiamata troppe volte

		if(visualizable_){
			std::cout<<"Tangential velocity -> central foot at contact: "<<slip_term_<<std::endl;
		}

		num_episode_++;
	};

 private:
  int gcDim_, gvDim_, nJoints_,timing_,fbk_counter_,n_campione_vel_,n_campione_pos_, index_imitation_;
  float alfa_z_, roll_init_, pitch_init_, yaw_init_; // initial orientation of prisma prisma walker
  double alfa_motor_offset_,beta_motor_offset_,gamma_motor_offset_, vx_int_,vy_int_,vz_int_,x_int_,y_int_,z_int_;
  Eigen::VectorXd m1_pos_;
  Eigen::VectorXd m2_pos_;
  int p_ = 0;int n_campioni_ = 5;
  dynamixel::PortHandler *portHandler_;
  dynamixel::PacketHandler *packetHandler_;
  uint8_t dxl_error_ = 0;
  int dxl_comm_result_ = COMM_TX_FAIL;
  Eigen::VectorXd azione_precedente_ = Eigen::VectorXd::Zero(3); 
  Eigen::VectorXd feedback_precedente_pos_ = Eigen::VectorXd::Zero(3); 
  Eigen::VectorXd feedback_precedente_vel_ = Eigen::VectorXd::Zero(3); 
  Eigen::VectorXd acc_precedente_ = Eigen::VectorXd::Zero(3); 
  //hebi::Lookup lookup_;
 // std::shared_ptr<hebi::Group> group_;
  double smoothing_factor_ = 0.06;
  Eigen::VectorXd pos_cmd_=Eigen::VectorXd::Zero(2);
  bool tester_real = false;
  raisim::Mat<3,3> rot_off_;
  raisim::Vec<4> quaternion_;
  bool first_time_ = true;
  int32_t dxl_present_position_ = 0;
  int32_t dxl_present_velocity_ = 0;
  Eigen::VectorXd m1_2_pos_feedback_ = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd m1_2_vel_feedback_ = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd filtered_acc_ = Eigen::VectorXd::Zero(3);
  /*hebi::GroupCommand cmd_ = hebi::GroupCommand(2);
  hebi::GroupFeedback fbk_ = hebi::GroupFeedback(2);*/
  Eigen::Quaternionf q_;
  Eigen::VectorXd campioni_acc_integrazione_x_ = Eigen::VectorXd::Zero(15);
  Eigen::VectorXd campioni_acc_integrazione_y_ = Eigen::VectorXd::Zero(15);
  Eigen::VectorXd campioni_acc_integrazione_z_ = Eigen::VectorXd::Zero(15);
  Eigen::VectorXd campioni_vel_integrazione_x_ = Eigen::VectorXd::Zero(15);
  Eigen::VectorXd campioni_vel_integrazione_y_ = Eigen::VectorXd::Zero(15);
  Eigen::VectorXd campioni_vel_integrazione_z_ = Eigen::VectorXd::Zero(15);
  double mean_value_x_ = 0.0;double mean_value_y_ = 0.0;double mean_value_z_ = 0.0;
  Eigen::VectorXd real_lin_acc_ = Eigen::VectorXd::Zero(3);
  /*hebi::Quaternionf real_orientation_=hebi::Quaternionf(1.0,0.0,0.0,0.0);
  hebi::Vector3f real_angular_vel_ = hebi::Vector3f(0.0,0.0,0.0);
  hebi::Vector3f real_linear_acc_ = hebi::Vector3f(0.0,0.0,0.0);*/
  Eigen::Matrix3d R_imu_base_frame_ = Eigen::Matrix3d::Zero(3,3);
  Eigen::Matrix3d real_orientation_matrix_ = Eigen::Matrix3d::Zero(3,3);
  Eigen::Vector3d rpy_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* prisma_walker;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget3_, vTarget_;
  const int terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, bodyAngularVel_real_;
  std::ofstream v_lin_sim_ = std::ofstream("v_lin_sim_40.txt");
  std::ofstream p_sim_ = std::ofstream("p_sim_40.txt");
  std::ofstream ori_sim_ = std::ofstream("ori_sim_40.txt");
  std::ofstream v_ang_sim_ = std::ofstream("v_ang_sim_40.txt");
  raisim::Vec<4> quat_;
  raisim::Mat<3,3> rot_;
  size_t foot_center_, footIndex_, foot_sx_ ,foot_dx_, contact_;
  raisim::CoordinateFrame footframe_,frame_dx_foot_,frame_sx_foot_;
  raisim::Vec<3> vel_, ang_vel_, vel_sx_, vel_dx_, ang_vel_sx_, ang_vel_dx_, footPosition_, footPosition_Sx_, footPosition_Dx_;
  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
 

  double slip_term_sxdx_, slip_term_;
  double previous_height_, max_height_, clearance_foot_ = 0;
  bool max_clearence_ = false;
  Eigen::Vector3d command_;

  std::chrono::duration<double, std::milli> elapsed_time_[3]; 
  std::chrono::duration<double, std::milli> swing_time_; 
  std::chrono::steady_clock::time_point begin_[3], lift_instant_, land_instant_;
  std::map<std::string,int> cF_ = {
    {"center_foot", 0},
    {"lateral_feet", 0},
  };
  int num_seq, num_seq_vel;
  Eigen::VectorXd joint_history_pos_, joint_history_vel_, current_action_;
  Eigen::VectorXd joint_history_pos_reshaped_, joint_history_vel_reshaped_;
  double H_ = 0.0;
  int num_episode_ = 0;
  double ang_vel_term_contact_ = 0;
  double previous_reward_, lin_reward_, ang_reward_ = 0;
 
  int previous_contact;
  int count_ = 0;
  double vel_rew_, mean_vel_rew_, swing_time_d;
  int NumberIteration_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

