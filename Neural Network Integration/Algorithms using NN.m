clc 
clear

%% Initialization
%initializing of decision variables
% nPV=0; % Initial No of pv Panels, One of the decision variables, it will be changed by the optimization Algorithm
% nwind=0; % No of wind turbines, the second decision variable, One of the decision variables, it will be changed by the optimization Algorithm
% V_max_proposed = 0; % current volume of water reservoir in m^3, third decision variable. It will be changed by the optimization Algorithm
% Q_T=0; % Turbine discharge rate in m^3/sec (turbine flow rate), the 4th decision variable. It will be changed by the optimization Algorithm
% Q_P=0; % Charging rate of the pump in m^3/sec, the 5th decision variable. It will be changed by the optimization Algorithm

% Load the saved neural network model
load('trained_model.mat');

% Num of Iterations
numIter = 20;
% row_vector = [bestSolution , elapsed_time];
Header = {'nPV','nWind','V_max_proposed','Q_T','Q_P','Cost','Elappsed Time','populationSize'};

% Specify the Excel file and sheet
excel_file = 'NN_Data.xlsx';

% Specify Sheet Name:
% sheet_name = 'FA_NN';
sheet_name = 'PSO_NN';
% sheet_name = 'GA_NN';
% sheet_name = 'SA_NN';

start_cell = 'A1';
xlswrite(excel_file, Header, sheet_name, start_cell);
PSO_Data = zeros(numIter,7);

multiplier = 1;
populationSize = 10*multiplier;
for iter = 1:numIter
    populationSize = populationSize + 200*multiplier; % Change population size with each iteration
    disp(populationSize);
%     out = FA(populationSize, net);
    out = PSO(populationSize, net);
%     out = GA(populationSize, net);
%     out = SA(net);
    
    PSO_Data(iter,:) = out;
    out = [out, populationSize];
    disp(["PSO Iteration: ", num2str(iter), '--', 'Fitness: ',out(6), '--', "Elappsed time: ", out(7)]);

    % Specify the starting cell dynamically based on the iteration
    start_row = iter + 1;  % A2 corresponds to row 2
    start_cell = ['A', num2str(start_row)];
    xlswrite(excel_file, out, sheet_name, start_cell);
end

[min_value , min_index] = min(PSO_Data(:,6));
fitness = PSO_Data(:,6);
Time = PSO_Data(:,7);
sum_fit = sum(fitness);
sum_time = sum(Time);


%KPIs
optimal_solution_PSO = PSO_Data(min_index,1:5); %1
optimal_fitness_PSO = min_value; %2
mean_fitness_PSO = sum_fit/numIter; %3

standard_deviation_PSO = 0;
for i = 1:numIter
    standard_deviation_PSO = (fitness(i) - mean_fitness_PSO)^2;
end
standard_deviation_PSO = sqrt(standard_deviation_PSO)/numIter; %4

computation_time_PSO = sum_time/numIter; %5

KPIs = [optimal_fitness_PSO; mean_fitness_PSO; standard_deviation_PSO; computation_time_PSO];

KPI_Header = {"Optimal_solution"; 'optimal_fitness'; 'mean_fitness'; 'standard_deviation'; 'computation_time'};
xlswrite(excel_file, KPI_Header, sheet_name, 'I1');
xlswrite(excel_file, optimal_solution_PSO, sheet_name, 'J1');
xlswrite(excel_file, KPIs, sheet_name, 'J2');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%%%%PLOTTING%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%%%%FUNCTIONS%%%%%%%%%%%%%%%%%%%%
%FA Function
function output = FA(populationSize, net)
%Initalize Parameters
% Start the timer
tic;
% populationSize = 20;
numGenerations = 100;
numDimensions = 5;
alpha = 1;   % Randomization Parameter
gamma = 1;   % absorption coefficient
beta = 1.5;  % Attraction Coefficient
delta = 0.5; % Randomization Parameter for Attractiveness

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
% Initialization of the decision variables within appropriate bounds
lowerBounds = [Min_pv, Min_wt, V_min, Q_T_min, Q_P_min]; % Minimum bounds for each variable
upperBounds = [Max_pv , Max_wt , V_max , Q_T_max ,Q_P_max ]; % Maximum bounds for each variable
variable_ranges = [lowerBounds;upperBounds];


%Initalize fireflies
fireflies = initialize_population_normal(populationSize,numDimensions,variable_ranges);
fireflies = [fireflies, zeros(populationSize, 1)]; %Augmenting a zero coloumn for evaluating the fitness value.


%Fitness Initializtion
% for i = 1:populationSize
%     nPV = fireflies(i, 1);
%     nwind = fireflies(i, 2);
%     V_max_proposed = fireflies(i, 3);
%     Q_T = fireflies(i, 4);
%     Q_P = fireflies(i, 5);
%     fireflies(i,6) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);   
% end
fireflies(:,6) =  sim(net, fireflies(:,1:5)')'; %TODO

[min_value, min_index] = min(fireflies(:,6)); % Find the minimum value and its index
CurrBest = fireflies(min_index,:);

%%MAINLOOP%%
for generation = 1:numGenerations
    
     % Apply random movement to enhance exploration
    fireflies = apply_random_movement(fireflies, alpha);
    
    %Clip the values to be within bounds
    for i = 1:numDimensions
            min_range = lowerBounds(i);
            max_range = upperBounds(i);
    
            % Clip values to be within the specified range
            fireflies(:, i) = max(min(fireflies(:, i), max_range), min_range);
    end

    %  Update light intensity based on objective function values
%     for i = 1:populationSize
%         nPV = fireflies(i, 1);
%         nwind = fireflies(i, 2);
%         V_max_proposed = fireflies(i, 3);
%         Q_T = fireflies(i, 4);
%         Q_P = fireflies(i, 5);
%         fireflies(i,6) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);   
%     end
    fireflies(:,6) =  sim(net, fireflies(:,1:5)')'; %TODO

    for i = 1:populationSize
        for j = 1:populationSize
            % Evaluate the attractiveness between fireflies i and j
            if(i == j)
                continue
            end
            if fireflies(j, 6) < fireflies(i, 6)
                % Calculate attractiveness based on distance
                attractiveness = calculate_attractiveness(distance(fireflies(i, 1:5), fireflies(j, 1:5),variable_ranges), beta, gamma, delta);
                % Move firefly i towards j with the calculated attractiveness
                fireflies(i, 1:5) = move_firefly(fireflies(i, 1:5), fireflies(j, 1:5), alpha, beta, attractiveness);
                %Evaluate new solutions and update light intensity
%                 for var = 1:numDimensions
%                     % Clip values to be within the specified range
%                     fireflies(i, var) = max(min(fireflies(i, var), upperBounds(var)), lowerBounds(var));
%                 end               
%                 fireflies(i,6) = evaluate_cost_function(fireflies(i, 1),fireflies(i, 2),fireflies(i, 3),fireflies(i, 4),fireflies(i, 5),net);
                fireflies(:,6) =  sim(net, fireflies(:,1:5)')'; %TODO
            end
        end
    end


[min_value, min_index] = min(fireflies(:,6)); % Find the minimum value and its index
CurrBest = fireflies(min_index,:);
 
    
end

bestSolution = CurrBest(1,:);

% Stop the timer
elapsed_time = toc;

output = [bestSolution, elapsed_time];


end

function population = initialize_population_normal(m, n, variable_ranges)
    % Initialize the positions of the particles in the swarm with a normal distribution.
    
    population = zeros(m, n);

    for i = 1:n
        min_range = variable_ranges(1, i);
        max_range = variable_ranges(2, i);
        
        % Set mean of the normal distribution within the range
        mean_val = (max_range + min_range) / 2;
        
        % Adjust standard deviation as needed
        std_dev = (max_range - min_range) / 4;

        % Generate random values from a normal distribution
        population(:, i) = normrnd(mean_val, std_dev, [m, 1]);

        % Clip values to be within the specified range
        population(:, i) = max(min(population(:, i), max_range), min_range);
    end
end

function distance_ij = distance(position_i, position_j, variable_ranges)
    % Euclidean distance between fireflies i and j with normalization
    normalized_position_i = (position_i - variable_ranges(1, :)) ./ (variable_ranges(2, :) - variable_ranges(1, :));
    normalized_position_j = (position_j - variable_ranges(1, :)) ./ (variable_ranges(2, :) - variable_ranges(1, :));
    distance_ij = norm(normalized_position_i - normalized_position_j);

%     distance_ij = norm(position_i - position_j);
end

function attractiveness = calculate_attractiveness(distance_ij, beta, gamma, delta)
    % Calculate attractiveness between fireflies i and j based on their light intensity and distance
    attractiveness = exp(-gamma * distance_ij) / (beta * distance_ij + delta);
end

function new_position = move_firefly(position_i, position_j, alpha, beta, attractiveness)
    % Move firefly i towards j with the calculated attractiveness
    new_position = position_i + beta * attractiveness * (position_j - position_i) + alpha*randn(size(position_i));
end

function fireflies = apply_random_movement(fireflies, alpha)
    % Input:
    %   fireflies: Matrix representing the positions of fireflies (each row is a firefly)
    %   alpha: Parameter controlling the step size of random movement

    % Get the number of fireflies and the dimensionality of the problem
    [num_fireflies, dimension] = size(fireflies);

    % Generate a random movement for each firefly
    random_movement = alpha * (rand(num_fireflies, dimension) - 0.5);

    % Apply the random movement to the fireflies
    fireflies = fireflies + random_movement;
end

%PSO Function
function output = PSO(populationSize, net)

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
% Initialization of the decision variables within appropriate bounds
lowerBounds = [Min_pv, Min_wt, V_min, Q_T_min, Q_P_min]; % Minimum bounds for each variable
upperBounds = [Max_pv , Max_wt , V_max , Q_T_max ,Q_P_max ]; % Maximum bounds for each variable
variable_ranges = [lowerBounds;upperBounds];

% PSO Parameters
% Start the timer
tic;
numIterations = 100;
% populationSize = 20;
numDimensions = 5;
w = 0.792; % Inertia
c1 = 1.4944; % Cognitive Factor
c2 = 1.4944; % Social Factor

swarm_position = initialize_swarm_positions_normal(populationSize,numDimensions,variable_ranges);
swarm_position = [swarm_position, zeros(populationSize, 1)]; %Augmenting a zero coloumn for evaluating the fitness value.


%Velocity Initializtion
velocity = zeros(populationSize,numDimensions);


%Fitness Initializtion
% for i = 1:populationSize
%     nPV = swarm_position(i, 1);
%     nwind = swarm_position(i, 2);
%     V_max_proposed = swarm_position(i, 3);
%     Q_T = swarm_position(i, 4);
%     Q_P = swarm_position(i, 5);
%     swarm_position(i,6) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);   
% end

swarm_position(:,6) =  sim(net, swarm_position(:,1:5)')'; %TODO

%Best Initializtion
pBest = swarm_position;

[min_value, min_index] = min(swarm_position(:,6)); % Find the minimum value and its index
nBest = swarm_position(min_index,:);
nBest = repmat(nBest, populationSize, 1);


%Saving the results for plotting 
%    Min_pv, Min_wt, V_min, Q_T_min, Q_P_min
plotted_nPV=zeros(1,numIterations);
plotted_nwind=zeros(1,numIterations);
plotted_V_max_proposed=zeros(1,numIterations);
plotted_Q_T=zeros(1,numIterations);
plotted_Q_P=zeros(1,numIterations);
plotted_cost=zeros(1,numIterations);

 
%%MAINLOOP%%
for iteration = 1:numIterations
% Calculate velocity
r1 = rand(populationSize, numDimensions);
r2 = rand(populationSize, numDimensions);
velocity = w.*velocity + c1.*r1.*(pBest(:,1:5) - swarm_position(:,1:5)) + c2.*r2.*(nBest(:,1:5) - swarm_position(:,1:5));
% disp(swarm_position(1,:));
swarm_position(:,1:5) = swarm_position(:,1:5) + velocity;
% disp(swarm_position(1,:));

%Clip the values to be within bounds
for i = 1:numDimensions
        min_range = lowerBounds(i);
        max_range = upperBounds(i);

        % Clip values to be within the specified range
        swarm_position(:, i) = max(min(swarm_position(:, i), max_range), min_range);
end
 
% Evaluate fitness
% for i = 1:populationSize
%     nPV = swarm_position(i, 1);
%     nwind = swarm_position(i, 2);
%     V_max_proposed = swarm_position(i, 3);
%     Q_T = swarm_position(i, 4);
%     Q_P = swarm_position(i, 5);
%     swarm_position(i,6) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);   
% end

swarm_position(:,6) =  sim(net, swarm_position(:,1:5)')'; %TODO
   
%Update Personal Best
for i = 1:populationSize
    if(swarm_position(i,6) < pBest(i,6))
        pBest(i,:) = swarm_position(i,:);
    end
end

%Update neighborhood Best
[min_value, min_index] = min(swarm_position(:,6)); % Find the minimum value and its index
if(swarm_position(min_index,6) < nBest(1,6))
    nBest = swarm_position(min_index,:);
    nBest = repmat(nBest, populationSize, 1);
end

      
% disp(['The Best in Generation: ',num2str(iteration) ,' - Fitness: ', num2str(nBest(1,6)),'----Decision variables:', num2str(nBest(1,1)),',',num2str(nBest(1,2)),',',num2str(nBest(1,3)),',',num2str(nBest(1,4)),',',num2str(nBest(1,5))]);
% disp(['---------------------------------']);
   
    
plotted_nPV(1,iteration) = nBest(1,1);
plotted_nwind(1,iteration) = nBest(1,2);
plotted_V_max_proposed(1,iteration) = nBest(1,3);
plotted_Q_T(1,iteration) = nBest(1,4);
plotted_Q_P(1,iteration) = nBest(1,5);
plotted_cost(1,iteration) = nBest(1,6);

    
end
bestSolution = nBest(1,:);
% disp(['Best Solution: ', num2str(bestSolution)]); % Display the final best solution 
elapsed_time = toc; % Stop the timer
% disp(['Elapsed Time: ', num2str(elapsed_time), ' seconds']); % Display the elapsed time

output = [bestSolution(1,1), bestSolution(1,2), bestSolution(1,3), bestSolution(1,4), bestSolution(1,5), bestSolution(1,6),elapsed_time(1,1)];


end %PSO Function end

%GA Function
function out = GA(populationSize, net)
% Genetic Algorithm with Elitism, CrossOver and Mutation
% GA Parameters
tic;
populationSize = 20;
numGenes = 5;
numGenerations = 100;
elitePercentage = 0.1;
crossoverPercentage = 0.6;
mutationPercentage = 0.3;
alpha = 0.7;

% Initialization
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


%Saving the results for plotting 
 plotted_nPV=zeros(1,numGenerations);
 plotted_nwind=zeros(1,numGenerations);
 plotted_V_max_proposed=zeros(1,numGenerations);
 plotted_Q_T=zeros(1,numGenerations);
 plotted_Q_P=zeros(1,numGenerations);
 plotted_cost=zeros(1,numGenerations);

 
% Initialization of the decision variables within appropriate bounds
lowerBounds = [Min_pv, Min_wt, V_min, Q_T_min, Q_P_min]; % Minimum bounds for each variable
upperBounds = [Max_pv , Max_wt , V_max , Q_T_max ,Q_P_max ]; % Maximum bounds for each variable
population = rand(populationSize, numGenes) .* (upperBounds - lowerBounds) + lowerBounds;
population = [population, zeros(populationSize, 1)]; %Augmenting a zero coloumn for evaluating the fitness value.

for generation = 1:numGenerations
% Evaluate fitness
  
fitnessValues = zeros(populationSize, 1);

for i = 1:populationSize
    nPV = population(i, 1);
    nwind = population(i, 2);
    V_max_proposed = population(i, 3);
    Q_T = population(i, 4);
    Q_P = population(i, 5);
    fitnessValues(i) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);
   
end
    population(:, 6)=fitnessValues;
    % Sort the matrix based on the fitness value which is the 6th coloum
    population = sortrows(population,6);



%     Min_pv, Min_wt, V_min, Q_T_min, Q_P_min
    
    
    % Select elite
    eliteCount = round(elitePercentage * populationSize);
    elite = population(1:eliteCount, :);
    
    %print the best fitness value and the corresponding decision variables 
   
  
    
    plotted_nPV(1,generation) = elite(1,1);
    plotted_nwind(1,generation) = elite(1,2);
    plotted_V_max_proposed(1,generation) = elite(1,3);
    plotted_Q_T(1,generation) = elite(1,4);
    plotted_Q_P(1,generation) = elite(1,5);
    plotted_cost(1,generation) = elite(1,6);

    % Perform crossover
    crossoverCount = round(crossoverPercentage * populationSize);
    crossoverPopulation =population(eliteCount+1:eliteCount+crossoverCount,:);
    
    % Assuming crossoverCount is the number of pairs of children to be produced
for i = 1:(crossoverCount/2)
    parent1 = elite(randi(eliteCount), :); %parent 1 is a random number from the elite
    %parent1 =crossoverPopulation(randi(crossoverCount), :); %parent 1 is a random number from the crossover
    %parent2 = crossoverPopulation(i, :);   %parent 2 is from the cross over population   
    parent2 =crossoverPopulation(randi(crossoverCount), :);
    % Crossover operation to produce two children
    child1 = alpha * parent1 + (1 - alpha) * parent2;
    child2 = (1 - alpha) * parent1 + alpha * parent2;
    
    %checking that the new generated childs lie within the boundary constraints 
    if(child1(1)>upperBounds(1))
        child1(1)=upperBounds(1);
    end
    if(child1(1)<lowerBounds(1))
        child1(1)=lowerBounds(1);
    end 
    if(child1(2)>upperBounds(2))
        child1(2)=upperBounds(2);
    end
    if(child1(2)<lowerBounds(2))
        child1(2)=lowerBounds(2);
    end 
        
    if(child1(3)>upperBounds(3))
        child1(3)=upperBounds(3);
    end
    if(child1(3)<lowerBounds(3))
        child1(3)=lowerBounds(3);
    end 
    if(child1(4)>upperBounds(4))
        child1(4)=upperBounds(4);
    end
    if(child1(4)<lowerBounds(4))
        child1(4)=lowerBounds(4);
    end 
    if(child1(5)>upperBounds(5))
        child1(5)=upperBounds(5);
    end
    if(child1(5)<lowerBounds(5))
        child1(5)=lowerBounds(5);
    end 
    
    
    if(child2(1)>upperBounds(1))
        child2(1)=upperBounds(1);
    end
    if(child2(1)<lowerBounds(1))
        child2(1)=lowerBounds(1);
    end 
    if(child2(2)>upperBounds(2))
        child2(2)=upperBounds(2);
    end
    if(child2(2)<lowerBounds(2))
        child2(2)=lowerBounds(2);
    end 
        
    if(child2(3)>upperBounds(3))
        child2(3)=upperBounds(3);
    end
    if(child2(3)<lowerBounds(3))
        child2(3)=lowerBounds(3);
    end 
    if(child2(4)>upperBounds(4))
        child2(4)=upperBounds(4);
    end
    if(child2(4)<lowerBounds(4))
        child2(4)=lowerBounds(4);
    end 
    if(child2(5)>upperBounds(5))
        child2(5)=upperBounds(5);
    end
    if(child2(5)<lowerBounds(5))
        child2(5)=lowerBounds(5);
    end 
            
    % Storing children in crossoverPopulation
    New_crossoverPopulation(2*i-1, :) = child1;
    New_crossoverPopulation(2*i, :) = child2;
end
%    %sorting the results from crossover for selecting the best half of crossovers 
%    for i = 1:(crossoverCount*2)
%     nPV = New_crossoverPopulation(i, 1);
%     nwind = New_crossoverPopulation(i, 2);
%     V_max_proposed = New_crossoverPopulation(i, 3);
%     Q_T = New_crossoverPopulation(i, 4);
%     Q_P = New_crossoverPopulation(i, 5);
%     crossover_fitnessValues(i) = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,w_COE,w_D_Load,w_LSPS);
%    end
%    New_crossoverPopulation(:, 6)= crossover_fitnessValues;
%    % Sort the cross over matrix based on the fitness value which is the 6th coloum
%    New_crossoverPopulation = sortrows( New_crossoverPopulation,6);
%    %take the best half from cross over based on their sorted fitness value
   crossoverPopulation=New_crossoverPopulation(1:crossoverCount,:);
   
% Perform mutation
mutationCount = round(mutationPercentage * populationSize);
mutationPopulation =population((eliteCount+crossoverCount+1):end,:); 
%noise on mutation

mutationPopulation= mutationPopulation+randn(mutationCount, numGenes+1) * 10;

%making sure that the resulting mutants lie within the boundary constraints
for(i=1:mutationCount)
    
    if(mutationPopulation(i,1)>upperBounds(1))
        mutationPopulation(i,1)=upperBounds(1);
    end
    if(mutationPopulation(i,1)<lowerBounds(1))
        mutationPopulation(i,1)=lowerBounds(1);
    end 
    if(mutationPopulation(i,2)>upperBounds(2))
        mutationPopulation(i,2)=upperBounds(2);
    end
    if(mutationPopulation(i,2)<lowerBounds(2))
        mutationPopulation(i,2)=lowerBounds(2);
    end 
        
    if(mutationPopulation(i,3)>upperBounds(3))
        mutationPopulation(i,3)=upperBounds(3);
    end
    if(mutationPopulation(i,3)<lowerBounds(3))
        mutationPopulation(i,3)=lowerBounds(3);
    end 
    if(mutationPopulation(i,4)>upperBounds(4))
        mutationPopulation(i,4)=upperBounds(4);
    end
    if(mutationPopulation(i,4)<lowerBounds(4))
        mutationPopulation(i,4)=lowerBounds(4);
    end 
    if(mutationPopulation(i,5)>upperBounds(5))
        mutationPopulation(i,5)=upperBounds(5);
    end
    if(mutationPopulation(i,5)<lowerBounds(5))
        mutationPopulation(i,5)=lowerBounds(5);
    end 
    
    
end    
% Create next generation
 
   newGeneration = [elite; crossoverPopulation; mutationPopulation];

   % Update population for the next generation
    
   population = newGeneration;
   

end


bestSolution = population(1, :);

elapsed_time = toc;
out = [bestSolution, elapsed_time];


end % Function GA END

function out = SA()

%initializing of decision variables
nPV=1000; % Initial No of pv Panels, One of the decision variables, it will be changed by the optimization Algorithm
nwind=1000; % No of wind turbines, the second decision variable, One of the decision variables, it will be changed by the optimization Algorithm
V_max_proposed = 10000; % current volume of water reservoir in m^3, third decision variable. It will be changed by the optimization Algorithm
Q_T=5; % Turbine discharge rate in m^3/sec (turbine flow rate), the 4th decision variable. It will be changed by the optimization Algorithm
Q_P=5; % Charging rate of the pump in m^3/sec, the 5th decision variable. It will be changed by the optimization Algorithm

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
% Simulated Annealing parameters
tic;
initial_temperature = 100; % Initial temperature
final_temperature = 1;  % Final temperature
iterations_per_temperature = 1; % Number of iterations at each temperature
beta_Simulated_Annealing=1; % No of degrees to be subtracted per each temperature change
maximum_iterations=round((initial_temperature-final_temperature)/beta_Simulated_Annealing);

%Saving the results for plotting

 plotted_nPV=zeros(1,maximum_iterations);
 plotted_nwind=zeros(1,maximum_iterations);
 plotted_V_max_proposed=zeros(1,maximum_iterations);
 plotted_Q_T=zeros(1,maximum_iterations);
 plotted_Q_P=zeros(1,maximum_iterations);
 plotted_cost=zeros(1,maximum_iterations);

 
% Initialize the decision variables within appropriate bounds
% Customize these variables and bounds based on your problem
nVariables = 5; % Number of decision variables

% Initial solution (random values but we try to initialize them with values near the optimal solution according to the paper)
initial_solution = [nPV,nwind,V_max_proposed,Q_T,Q_P]; 
best_solution=initial_solution;% the best decision variables are at the beginning 
variable_min = [Min_pv, Min_wt, V_min, Q_T_min, Q_P_min]; % Minimum bounds for each variable
variable_max = [Max_pv , Max_wt , V_max , Q_T_max ,Q_P_max ]; % Maximum bounds for each variable

current_cost=evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);
best_cost=current_cost; %initial value of the cost function is the best so far

%Here Begins the Simulated Annealing Algorithm
% Simulated Annealing algorithm
temperature = initial_temperature;

for outer_iteration = 1:maximum_iterations
       for inner_iteration = 1:iterations_per_temperature
        neighbor_solution = initial_solution + initial_solution .* (0.2 * randn(1, nVariables));
        neighbor_solution = max(min(neighbor_solution, variable_max), variable_min);
        nPV=neighbor_solution(1); 
        nwind=neighbor_solution(2); 
        V_max_proposed =neighbor_solution(3) ; 
        Q_T=neighbor_solution(4); 
        Q_P=neighbor_solution(5);
       



%These parameters are initial values, hoever they should be the values computed by the algortithm itself
neighbor_cost = evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);
delta_cost = neighbor_cost - best_cost;
if delta_cost<0
  best_solution = neighbor_solution;
  best_cost = neighbor_cost;
else
    %accepting a new solution
 if rand() < exp(-delta_cost / temperature)
     initial_solution = neighbor_solution;
     nPV=neighbor_solution(1); 
     nwind=neighbor_solution(2); 
     V_max_proposed =neighbor_solution(3) ; 
     Q_T=neighbor_solution(4); 
     Q_P=neighbor_solution(5);
current_cost=evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net);
if current_cost<best_cost
    best_cost=current_cost;
    best_solution=initial_solution;
    
 end
 end
    
end
       end
       temperature = initial_temperature -beta_Simulated_Annealing;
    % Plot the results
   plotted_nPV(outer_iteration)=round(best_solution(1));
   plotted_nwind(outer_iteration)=round(best_solution(2));
   plotted_V_max_proposed(outer_iteration)=best_solution(3);
   plotted_Q_T(outer_iteration)=best_solution(4);
   plotted_Q_P(outer_iteration)=best_solution(5);
   plotted_cost(outer_iteration)=best_cost;
    
   elapsed_time = toc;
   out = [best_solution(1), best_solution(2) , best_solution(3), best_solution(4), best_solution(5), best_cost, elapsed_time];

end

end

function swarm_positions = initialize_swarm_positions_normal(m, n, variable_ranges)
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

function current_cost=evaluate_cost_function(nPV,nwind,V_max_proposed,Q_T,Q_P,net)

inputs = [nPV;nwind;V_max_proposed;Q_T;Q_P];
current_cost = sim(net, inputs);

end



