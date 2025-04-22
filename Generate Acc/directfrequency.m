% StaBIL manual
% Tutorial: dynamic analysis: modal superposition: transform to f-dom
% Units: m, N
% Assembly of M, K and C
Dynamic_analysis;
[phi,omega]=eigfem(K,M); % Calculate eigenmodes and eigenfrequencies
xi=0.07; % Damping ratio
nModes=length(K); % OR length(K)-size(Constr,1)
C=M.'*phi(:,1:nModes)*diag(2*xi*omega(1:nModes))*phi(:,1:nModes).'*M;
% Modal -> full damping matrix C

L=L_span;
% Live load data -- see also 'trainload'
% 
% I	4.76	1.350	7.62	3.70	7.62	20.00
% II	5.40	1.385	10.30	3.80	10.30	26.00
% III	7.00	1.85	11.50	3.85	11.50	30.00

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
% Sampling parameters: time domain
T = (DTBB + L + DTBA + LT)/V; % Time window [s]
N=fix(T/dt); % Number of samples
t=(0:N-1)*dt; % Time axis (samples)
t1= DTBB/V; % [s] truck/train enter (firsr axle)
t2 = (DTBB + L + LT)/V; % [s] truck/train leave (last axle)
% Sampling parameters: frequency domain
F=1/dt; % Sampling frequency [Hz]
df=1/T; % Frequency resolution
f=[0:fix(N/2)-1]*df; % Positive frequencies corresponding to FFT [Hz]
Omega=2*pi*f; % [rad/s] excitation frequency content
% Excitation: transfer PLoad vector to nodalforce vector
% (seldof --> all DOF)
Pnodal = zeros(size(DOF,1),N);
for itime = 1:N
Pnodal(:,itime) = nodalvalues(DOF,seldof,PLoad(:,itime));
end
% Transfer nodal force vector time history to frequency domain
Q = zeros(size(DOF,1),fix(N/2)); % keep positive frequency ONLY
for indof = 1:size(DOF,1)
temp = fft(Pnodal(indof,:));
Q(indof,:) = temp(1:fix(N/2));
end
Pd = Q;
% Solution for each frequency
Ud=zeros(size(Pd));
for k=1:N/2
Kd=-Omega(k)^2*M+Omega(k)*i*C+K;
Ud(:,k)=Kd\Pd(:,k);
end
% F-dom -> t-dom
Ud=[Ud, zeros(length(K),1), conj(Ud(:,end:-1:2))];
u=ifft(Ud,[],2); % Nodal response (nDOF * N)
% Figures
figure;
c=selectdof(DOF,[2.03; 6.03; 7.03; 108.03; 11.03]);
plot(t,c*u);
title('Nodal response (direct method in f-dom)');
xlabel('Time [s]');
ylabel('Displacement [m]');
legend('2.03','6.03','7.03','108.03','11.03');
% % Movie
% figure;
% animdisp(Nodes,Elements,Types,DOF,u);
% Display
disp('Maximum nodal response 2.03; 6.03; 7.03; 108.03; 11.03');
disp(max(abs(c*u),[],2));