% direct time integration
Dynamic_analysis;
L=L_span;
% Eigenvalue analysis
nMode = 6; % Number of modes to take into account
[phi,omega]=eigfem(K,M); % Calculate eigenmodes and eigenfrequencies
xi=0.07; % Constant modal damping ratio

nModes=length(K);
C=M.'*phi(:,1:nModes)*diag(2*xi*omega(1:nModes))*phi(:,1:nModes).'*M;
% Modal -> full damping matrix C
% Truck/Train data
% P = [ -47600 -76200 -76200; % P1 P2 P3
% 0 3.70 1.35]; % 0 l2 l3

% P = [ -54000 -103000 -103000; % P1 P2 P3
% 0 3.80 1.385]; % 0 l2 l3

P = [ -70000 -115000 -115000; % P1 P2 P3
0 3.85 1.85]; % 0 l2 l3
DTBB =10; % [m] Distance train/truck to bridge before (front axle)
DTBA =30; % [m] Distance train/truck to bridge after (rear axle)
V = 60*1000/3600; % km/h --> [m/s] Velocity
LT = sum(P(2,:)); % [m] Length of train/truck

seldof=[
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.04;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.01;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.05;
[21;22;23;24;25;26;27;28;29;30;31;32;41;42;43;44;45;46;47;48;49;50;51;52]+0.02];

dt=0.001; % Time step/resolution
PLoad = trainload(P,L,DTBB,DTBA,V,dt,seldof,Nodes); % [seldof x N samples]

% Sampling parameters
T = (DTBB + L + DTBA + LT)/V; % Time window
N=fix(T/dt); % Number of samples
t=(0:N-1)*dt; % Time axis
t1= DTBB/V; % train enter
t2 = (DTBB + L + LT)/V; % train leave (last axle)

% Excitation
pm = zeros(size(DOF,1),N);
for itime = 1:N
pm(:,itime) = nodalvalues(DOF,seldof,PLoad(:,itime));
end
%% Direct time integration - trapezium rule
alpha=1/4;
delta=1/2;
theta=1;
u=wilson(M,C,K,dt,pm,[alpha delta theta]);

% Figures
figure;
c=selectdof(DOF,[2.03; 6.03; 7.03; 108.03; 11.03]);

plot(t,c*u);
title('Nodal response (direct time integration)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Nodal displacements [m]');
grid on;
legend('2.03','6.03','7.03','108.03','11.03','Location','SouthEast');

% % Movie
% figure;
% animdisp(Nodes,Elements,Types,DOF,u);

% Display
disp('Maximum nodal response 2.03; 6.03; 7.03; 108.03; 11.03');

disp(max(abs(c*u),[],2));
