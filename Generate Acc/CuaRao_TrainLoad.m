% Lexuanthang.official@gmail.com,Lexuanthang.official@outlook.com
% Le Xuan Thang, 2023
% Tutorial: dynamic analysis: direct method: frequency domain
% Units: m, N
CuaRaoBridge;
L=L_span;

% Train
P = [ -10000 -15000 -15000; % P1 P2 P3
0 5 10]; % 0 l2 l3

DTBB =30; % [m] Distance train/truck to bridge before (front axle)
V = 150*1000/3600; % km/h --> [m/s] Velocity
LT = sum(P(2,:)); % [m] Length of train/truck

seldof=reprow([600],1,8,[1])+0.03;
seldof = seldof(:);
dt=0.002; % Time step/resolution
t1= DTBB/V; % [s] truck/train enter (firsr axle)
t2 = (DTBB + L + LT)/V; % [s] truck/train leave (last axle)
t3 = 2;

% Sampling parameters: time domain
T = ((DTBB + L + LT)/V) + 100; % Time window [s]
N=fix(T/dt); % Number of samples 10739
t=(0:N)*dt; % Time axis (samples)

% Sampling parameters: frequency domain
F=1/dt; % Sampling frequency [Hz]
df=1/T; % Frequency resolution
f=(0:fix(N/2)-1)*df; % Positive frequencies corresponding to FFT [Hz]
Omega=2*pi*f; % [rad/s] excitation frequency content

% Excitation: transfer PLoad vector to nodalforce vector
% (seldof --> all DOF)
startInterval = 2;
nloop = 7;
Pulse = -500; %N
gap = 10;
PLoad = trainload(P,L,DTBB,V,dt,seldof,Nodes,startInterval,Pulse,T, nloop, gap,f0); % [seldof x N samples]
% PLoad = awgn(PLoad,4,'measured','linear');
figure;
plot(PLoad(1,:))
% Eigenvalue analysis
nMode = 6; % Number of modes to take into account
[phi,omega]=eigfem(K,M,nMode); % Calculate eigenmodes and eigenfrequencies
xi=0.07; % Constant modal damping ratio

Pnodal = zeros(size(DOF,1),N);
for itime = 1:N
    Pnodal(:,itime) = nodalvalues(DOF,seldof,PLoad(:,itime));
end

% Modal excitation
Pm_ = phi.'*Pnodal; % [DOF,nMode] x [DOF, N samples]

% Transfer nodal force vector time history to frequency domain
Q = zeros(nMode,fix(N/2)); % keep positive frequency ONLY
for inMode = 1:nMode
    temp = fft(Pm_(inMode,:));
    Q(inMode,:) = temp(1:fix(N/2));
end

Pm = Q;

% Modal analysis: calc. the modal transfer functions and the modal disp.
[X,H]=msupf(omega,xi,Omega,Pm); % Modal response, positive freq (nMode * N/2)
% F-dom -> t-dom [inverse Fourier transform]
X = [X, zeros(nMode,1), conj(X(:,end:-1:2))];
x = ifft(X,[],2); % Modal response (nMode * N)

% Modal displacements -> nodal displacements
u=phi*x; % Nodal response (nDOF * N)
% Figures
figure;
subplot(2,2,1);
if length(t) ~= length(PLoad)
    plot(t(1:end-1),PLoad(1,:),'.-');
else
    plot(t,PLoad(1,:),'.-');
end
xlim([0 T])
ylim('auto');
title('Excitation time history');
xlabel('Time [s]');
ylabel('Force [N]');

grid on

subplot(2,2,2);
plot(f,abs(Q([1,2,3,4,5,6],:))/F,'.-');
title('Excitation frequency content');
xlabel('Frequency [Hz]');
ylabel('Force [N/Hz]');
xlim([0 3])
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);
grid on

subplot(2,2,3);
plot(f,abs(H([1,2,3,4,5,6],:)),'.-');
title('Modal transfer function');
xlabel('Frequency [Hz]');
ylabel('Displacement [m/N]');
xlim([0 10])
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);grid on

subplot(2,2,4);
plot(f,abs(X(:,1:fix(N/2)))/F,'.-');
xlim([0 3])
title('Modal response');
xlabel('Frequency [Hz]');
ylabel('Displacement [m kg^{0.5}/Hz]');grid on
legend([repmat('Mode ',6,1) num2str([1,2,3,4,5,6].')]);

figure;
plot(t(1:length(x)),x);
title('Modal response (calculation in f-dom)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Displacement [m kg^{0.5}]');
legend([repmat('Mode ',nMode,1) num2str((1:nMode).')]);

figure;
d = reprow([101,301],1,6,[1,1])+0.03;
c = selectdof(DOF,d(:));
% u_o = awgn(c*u,1000,'measured','linear');
u_o = c*u;
plot(t(1:1:length(x)),u_o);
title('Nodal response (computed in the frequency domain)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Displacement [m]');grid on
legend(num2str(d(:)));

acceleration = zeros(size(u_o));
for i = 1:size(u_o,1)
    acceleration(i,:) = displacementToAcceleration(u_o(i,:), dt);
end
figure;
plot(t(1:1:length(x)),acceleration(1,:));
title('acceleration(m/s2)');
xlabel('Time [s]');
xlim([0 T])
ylabel('Acceleration [m]');grid on
legend(num2str(d(:)));

% Movie
% figure;
% animdisp(Nodes,Elements,Types,DOF,u);
% Display
disp('Maximum modal response');
disp(max(abs(x),[],2));
disp('Maximum nodal response 2.03; 6.03; 7.03; 108.03; 11.03');
disp(max(abs(c*u),[],2));

