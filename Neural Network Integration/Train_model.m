clc 
clear
%Cost function weights and cost function evaluation
w_COE=0.9;
w_D_Load=0.02;
w_LSPS=0.08;
%% Initialization
%initializing of decision variables
nPV=0; % Initial No of pv Panels, One of the decision variables, it will be changed by the optimization Algorithm
nwind=0; % No of wind turbines, the second decision variable, One of the decision variables, it will be changed by the optimization Algorithm
V_max_proposed = 0; % current volume of water reservoir in m^3, third decision variable. It will be changed by the optimization Algorithm
Q_T=0; % Turbine discharge rate in m^3/sec (turbine flow rate), the 4th decision variable. It will be changed by the optimization Algorithm
Q_P=0; % Charging rate of the pump in m^3/sec, the 5th decision variable. It will be changed by the optimization Algorithm

%decision variables minimum constraints
V_min = 500;
Min_pv=10;
Min_wt=10;
Q_T_min=1;
Q_P_min=1;
%decision variables maximum constraints
Max_pv=2000;
Max_wt=1000;
Q_T_max=10;
Q_P_max=10;
V_max=20000;


n = 1000; % Number of data points
numInputs = 5;
rng(42); % Set the seed for reproducibility

% Initialization of the decision variables within appropriate bounds
lowerBounds = [Min_pv, Min_wt, V_min, Q_T_min, Q_P_min]; % Minimum bounds for each variable
upperBounds = [Max_pv , Max_wt , V_max , Q_T_max ,Q_P_max ]; % Maximum bounds for each variable
variable_ranges = [lowerBounds;upperBounds];

inputs = gen_inputs(n,numInputs,variable_ranges);

%Fitness Initializtion
targets = zeros(n,1);
for i = 1:n
    nPV = inputs(i, 1);
    nwind = inputs(i, 2);
    V_max_proposed = inputs(i, 3);
    Q_T = inputs(i, 4);
    Q_P = inputs(i, 5);
    targets(i,1) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,w_COE,w_D_Load,w_LSPS);   
end


% Define the split ratios
trainRatio = 0.7; % 70% for training
valRatio = 0;   % 15% for validation
testRatio = 0.3;  % 15% for testing

% Divide the data
[trainInd, valInd, testInd] = dividerand(n, trainRatio, valRatio, testRatio);

% Create training, validation, and testing sets
trainInputs = inputs(trainInd, :);
trainTargets = targets(trainInd, :);

valInputs = inputs(valInd, :);
valTargets = targets(valInd, :);

testInputs = inputs(testInd, :);
testTargets = targets(testInd, :);

% Transpose 
inputs = inputs';
targets = targets';
testInputs = testInputs';
testTargets = testTargets';

% Create a neural network
net = feedforwardnet(10); % Replace with your desired network architecture

% Set optimization algorithm and error calculation type
net.trainFcn = 'trainlm'; % Levenberg-Marquardt backpropagation
net.performFcn = 'mse';    % Mean Squared Error

% Set training parameters, including validation data
net.divideFcn = 'dividerand'; % Use random data division
net.divideParam.trainRatio = trainRatio;
net.divideParam.valRatio = valRatio;
net.divideParam.testRatio = testRatio;

% Train the neural network
% net = train(net, inputs, targets, 'useParallel', 'yes'); 
net = train(net, inputs, targets); 

% Test the neural network
numPoints = 50;
model_outputs = sim(net, testInputs(:,1:numPoints));
outputs = testTargets(:,1:numPoints);

% Save the trained neural network model
save('trained_model.mat', 'net');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
plot(outputs, 'b', 'LineWidth', 1, 'DisplayName', 'Actual Data');
hold on
plot(model_outputs, 'ro', 'MarkerSize', 8, 'DisplayName', 'Model Predictions');
xlabel('Data Point');
ylabel('Value');
title('Actual vs. Predicted Values');
legend('show');
grid on;
% Evaluate performance or do further analysis as needed

%%
%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%
function swarm_positions = gen_inputs(m, n, variable_ranges)
    % Initialize the positions of the particles in the swarm with a normal distribution.
    
    swarm_positions = zeros(m, n);

    for i = 1:n
        min_range = variable_ranges(1, i);
        max_range = variable_ranges(2, i);
        
        % Set mean of the normal distribution within the range
        mean_val = (max_range + min_range) / 2;
        
        % Adjust standard deviation as needed
        std_dev = (max_range - min_range) / 4;

        % Generate random values from a normal distribution
        swarm_positions(:, i) = normrnd(mean_val, std_dev, [m, 1]);

        % Clip values to be within the specified range
        swarm_positions(:, i) = max(min(swarm_positions(:, i), max_range), min_range);
    end
end

function [V_t_minus_1,  ES_T, E_def_t] = calculate_volume(V_t_minus_1, V_max, V_min, EB, Q_pump, Q_dis_t, EPump_t, EH_t)
    
    if(EB>0)
      if(V_t_minus_1<V_max)
          if(Q_pump<=V_max-V_t_minus_1)
              V_t_minus_1 = (V_t_minus_1 + ((Q_pump*31*24*60*60) - (Q_dis_t*31*24*60*60)));
          else
              V_t_minus_1 = V_max; 
          end
      else
          V_t_minus_1 = V_t_minus_1;
      end
    else
      if(V_t_minus_1>V_min)
          if(Q_dis_t<=V_t_minus_1-V_min)
              V_t_minus_1 = (V_t_minus_1 + ((Q_pump*31*24*60*60) - (Q_dis_t*31*24*60*60)));
          else
              V_t_minus_1 = V_min;
          end
      else
          V_t_minus_1 = V_t_minus_1;
      end
    end
    ES_T = calculate_ES_t(EB, EPump_t);
    E_def_t = calculate_E_def_t(EB, EH_t);
    
end

function EPV_t = calculate_EPV_t(nPV, PPVV, eta_PV, eta_INV, eta_Wire, Irad_t, Inom, betaT, TC_t, TC_nom)
    EPV_t = [0,0,0,0,0,0,0,0,0,0,0,0];
    tmp = 0;
    for c = 1:12
       tmp = nPV * PPVV * eta_PV * eta_INV * eta_Wire * (Irad_t(c))*(1/Inom) * (1 - betaT * (TC_t(c) - TC_nom));
       EPV_t(c) = tmp*31;
    end
    
end

function PWT = calculate_PWT(u, u_cut_in, u_rated, u_cut_off, nwind, etawind, PR_WT)
    u_cut_in_squared = u_cut_in^2;
    u_rated_squared = u_rated^2;

    if u < u_cut_in
        PWT = 0;
    elseif u_cut_in < u && u < u_rated
        PWT = (nwind * etawind * PR_WT * (u^2 - u_cut_in_squared)) / (u_rated_squared - u_cut_in_squared);
    elseif u_rated < u && u < u_cut_off
        PWT = nwind * etawind * PR_WT;
    else
        PWT = 0;
    end
end


function h_add = calculate_h_add(V_t_minus_1, area)
 % This function takes as parameter the current volume of the reservoir
 % and the area of reservoir and computes the current reservoir added head

h_add = V_t_minus_1 / area;

end

function EH_t = calculate_EH_t(V_t_minus_1, Q_T, etaT, etaWP, rho, g, h_add, h3,EB)
    EB = sum(EB);
    EH_t = min(min(V_t_minus_1 / 3600, Q_T) * etaT * etaWP * rho * g * (h_add + h3), abs(EB));
end


function EB = calculate_EB(EPV, EWT, ED)
    EB = EPV + EWT - ED;
end



function EPump_t = calculate_EPump_t(V_max, V_t_minus_1, Q_P, etap, etaWP, rho, g, h_add, h3,EB)
    EPump_t = min(min((V_max - V_t_minus_1) / 3600, Q_P) * etap * etaWP * rho * g * (h_add + h3), abs(EB));
end


function Q_dis_t = calculate_Q_dis_t(EH_t, etaT, etaWP, rho, g, h_add, h3)
    Q_dis_t = EH_t / (etaT * etaWP * rho * g * (h_add + h3));
end

function E_def_t = calculate_E_def_t(EB, EH_t)
    E_def_t = abs(EB + EH_t);
end

function Q_pump_t = calculate_Q_pump_t(EPump_t, etap, etaWP, rho, g, h_add, h3)
    Q_pump_t = EPump_t / (etap * etaWP * rho * g * (h_add + h3));
end

function ES_t = calculate_ES_t(EB_t, EPump_t)
    ES_t = EB_t - EPump_t;
end


function LPSP =  calculate_LPSp(pload,EPV, EWT, EH_t)
    %I need to replicate the pload for the number of hours in the year
   tmp1 = sum(pload);
   tmp = 0;
   add=0;
   for c = 1:365
        tmp = tmp + tmp1;
        add = add+(tmp1-(EPV + EWT + EH_t));
   end
    LPSP = add / tmp;
    if(LPSP<0)
         LPSP=0;
    end
       
end


function Dload = calculate_Dload(EPV, EWT, pload)
    N = length(pload);
    sum_squares = 0;
    mean_pload = mean(pload);
    tmpEPV = mean(EPV);
    for i = 1:N
        sum_squares = sum_squares + (tmpEPV + EWT - pload(i))^2;
    end

    Dload = sqrt(sum_squares / N) / mean_pload;
end


function [Cann_cap_i, sum_CRF] = calculate_annual_capital_cost(Ccap_i, r, Mi)
    n = length(Ccap_i);
    CRF = zeros(1, n);
    Cann_cap_i = zeros(1, n);
   
    for i = 1:n
      % CRF(i) = (r * (1 + r)^Mi(i)) / ((1 + r)^Mi(i) - 1);
        CRF(i) = calculate_crf(r, Mi(i));
        Cann_cap_i(i) = Ccap_i(i) * CRF(i);
    end
    Cann_cap_i = sum(Cann_cap_i);
    sum_CRF=sum(CRF);
end


function Co_and_m = operating_and_maintenance_cost(Cpvo_and_m, tpv, Cwto_and_m, twt, Chydroo_and_m, thydro, Cpumpo_and_m, tpump)
    Co_and_m = Cpvo_and_m * tpv + Cwto_and_m * twt + Chydroo_and_m * thydro + Cpumpo_and_m * tpump;
end


function Crep = calculate_replacement_cost(Ccap_i, Msys, Mi)
     Crep = (Ccap_i .* (Msys - Mi)) ./ Mi;
     Crep = sum(Crep);
end


function Cann_tot = calculate_annual_total_cost(Cann_cap, Camn_rep, Cann_O_M)
    Cann_tot = Cann_cap + Camn_rep + Cann_O_M;
   
end

function NPV = calculate_net_present_value(Cann_tot, CRF)
    NPV = Cann_tot / CRF;
end

function crf = calculate_crf(r, M)
    crf = (r * (1 + r)^M) / ((1 + r)^M - 1);
end

function COE = calculate_cost_of_energy(Cann_tot, pload)
   tmp1 =sum(pload);
   tmp = 0;
   for c = 1:365
        tmp = tmp + tmp1;
   end
   %pload=repmat(pload, 1, 365)
   COE = sum(Cann_tot) / tmp;
   
end

function min_f = minimize_objective_function(COE,D_load,LPSP,w_COE,w_D_Load,w_LSPS)
    min_f = w_COE * COE + w_D_Load * D_load + w_LSPS * LPSP;
end
function current_cost=evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,w_COE,w_D_Load,w_LSPS)
%PV Module parameters
PPVV=260; % Installed capacity of each PV module in watts
tpv=219000; % No. of hours in the year the pv panel should ideally work
%efficiencies are considered in ideal case
eta_PV=1; % conversion effiency of pv modules
eta_INV=1; % Inverter effiency of pv modules
eta_Wire=1; % Wire effiency of pv modules
Irad_t= (1/24)*[3200, 4200, 5600,6800,7600,8200,7600,7300,6200,4900,3500,3000]; % Current Ambient Solar Radiation Intensity in Watt/m^2 per Hour, it was 5.69KWH/m^2 per day but we changed
Inom=1000; % Intensity of Solar Radiation under Standard Conditions in Watt/m^2 per day,  
betaT=0.0045; % Temperature coefficient of power of the PV Module
TC_t= [35, 40, 45, 58, 70, 75, 80, 80, 75, 75, 65, 55]; % Current Cell temperature
TC_nom=25; % Cell temperature in celsius under standard conditions of Operation
% Wind Turbine Parameters
u=6.72; % Current wind speed m/sec
u_cut_in= 2.5; % Cut in wind speed which above it, the wind turbine can operate in m/sec
u_rated=12; % wind speed at rated power in m/sec
u_cut_off=25; % Cut-off wind speed after which the wind turbine must be shut down for safety reasons
etawind=1; % efficiency of wind system
PR_WT=30000; %  rated power of wind turbine in Watt
twt=175200; % No. of hours in the year the wind turbine should ideally work


% Water Reservoir Parameters
V_t_minus_1 = 0; % previous volume of water reservoir in m^3
V_min = 500;

area = 250; % The area of reservoir on m^2


%Turbine Generating mode parameters
etaT=1; % The efficiency of the turbine-generator
etaWP=1; %The efficieny of the pipeline
rho= 1000; %Water density
g=9.8; % Acceleration due to gravity
h3=80; % The offset between the lake and the ground of the reservoir in (m)
ED=410000; % Peak load demand in Watt
Chydroo_and_m=0.01*1000; % water turbine operating and maintence percentage in the year 1% of the year should be scheduled for maintenance
thydro=87600; % No. of hours in the year the turbine should ideally work

%Pump mode parameters
Q_pump = 0;
Q_dis_t = 0;
EPump_t = 0;
etap=1;
Cpumpo_and_m=0.01*10749; %  Pump operating and maintence percentage in the year 1% of the year should be scheduled for maintenance
tpump=87600; % No. of hours in the year the pump should ideally work

% the average load power demand per each hour in the day in watt
pload=10^3*[210,212,215,217,220,225,235,250,260,270,290,300,325,350,325,325,355,375,355,300,290,260,255,240]; 

% Annual capital cost parameters
r=0.06; % Interest rate
Mi=[25,20,10,10,25]; %Life time of each subsystem, PV, wind turbine,water turbine ,pump and reservoir respectively

%replacement cost parameters
Msys=25; % The life time of the whole hybrid system, 25 years
%Initialzing the cost function


%power coming pv panel,  this is an initial value in watts
EPV=calculate_EPV_t(nPV,PPVV,eta_PV,eta_INV,eta_Wire,Irad_t,Inom,betaT,TC_t,TC_nom);

%Power generated from the wind turbine,  this is an initial value in watts
EWT=calculate_PWT(u, u_cut_in, u_rated, u_cut_off, nwind, etawind, PR_WT);


% Energy balance, this is an initial value
EB = calculate_EB(EPV, EWT, ED);


ES_t = zeros(1,12);
E_def_t = zeros(1,12);
EH_t = zeros(1,12);
for i =  1:12
    h_add=calculate_h_add(V_t_minus_1, area);
    [V_t_minus_1,  ES_T, E_def_t] = calculate_volume(V_t_minus_1, V_max_proposed, V_min, EB(i), Q_pump, Q_dis_t, EPump_t, EH_t(i));
    EPump_t=calculate_EPump_t(V_max_proposed, V_t_minus_1, Q_P, etap, etaWP, rho, g, h_add, h3,EB(i));
    Q_pump=calculate_Q_pump_t(EPump_t, etap, etaWP, rho, g, h_add, h3);
    EH_t(i)=calculate_EH_t(V_t_minus_1, Q_T, etaT, etaWP, rho, g, h_add, h3,EB(i));
    Q_dis_t=calculate_Q_dis_t(EH_t(i), etaT, etaWP, rho, g, h_add, h3);
    E_def_t(i) = E_def_t;
    ES_t(i) = ES_T;
end
EB;
EPump_t;
EH_t;
[ psg_min , pgs_max ] = bounds( ES_t );


%surplus energy pushed into the governmental national network, when the energy produced by pv panels and wind turbines exceeds the load demand while the reservoir tank is full  
ES_t;


% Energy diffiency when the load demand is more than the energy supplied by the pv panel and wind turbine, the variable the decides wether we are in pumping or generating mode
E_def_t;


%LPSP
% Loss of power supplied probablity , which measures the probability of insufficient operation of the hybrid system when it fails
% to meet the load requirements, at that time we take energy from the national government
LPSP= calculate_LPSp(pload,EPV, EWT, EH_t); %FIX EH_t->v_t later
LPSP = mean(LPSP);

%Dynamic load is a factor which means that we should try to produce energy
%which is near the demand value so as not to either get extra energy from
%the government not have very exess amount of energy in which require great
%amount of units with very high initial capital cost.
Dload = calculate_Dload(EPV, EWT, pload);


%initial capital investment cost of intsallation of each subsystem,this is function of the number of installed unit, however, we only have
%one turbine and one pump,cost/m^3
Ccap_i=[112*nPV,58564.79*nwind,1000,10749,170*V_max_proposed];


%annual capital cost of each component
[ Cann_cap_i, sumCRF ]=calculate_annual_capital_cost(Ccap_i, r, Mi);  


%operating and maintenance cost parameters
Cpvo_and_m=0.01*112*(nPV); %pv panel operating and maintence percentage in the year 1% of the year should be scheduled for maintenance
Cwto_and_m= 0.03*58564.79*(nwind); % wind turbine operating and maintence percentage in the year 3% of the year should be scheduled for maintenance
Co_and_m = operating_and_maintenance_cost(Cpvo_and_m, tpv, Cwto_and_m, twt, Chydroo_and_m, thydro, Cpumpo_and_m, tpump);
Crep = calculate_replacement_cost(Ccap_i, Msys, Mi); % vector containing the replacement cost of each subsystem , respectively


%Parameters of total anual cost
Cann_tot = calculate_annual_total_cost(Cann_cap_i, Crep, Co_and_m);


%Net present cost
NPV=calculate_net_present_value(Cann_tot, sumCRF);


%Cost of Energy
COE=calculate_cost_of_energy(Cann_tot, pload);



%These parameters are initial values, hoever they should be the values computed by the algortithm itself
current_cost = minimize_objective_function(COE, Dload,LPSP,w_COE,w_D_Load,w_LSPS);

end


