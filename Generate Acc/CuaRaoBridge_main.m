%__________________________________________________________________________
% CUA RAO BRIDGE source codes version 1.0
%
%  Developed in MATLAB R2022
%
%  Author and programmer: Le Xuan Thang / Viktor
%
%         e-Mail: lxt1021997lxt@gmail.com
%                 thang4201097sdh@lms.utc.edu.vn
%
%       ORCID: 0000-0002-9911-3544
%
%   Main paper:
%   DOI:
%__________________________________________________________________________
% clear
% close all
% clc

% Element types -> {EltTypID EltName}
Types=  {1 'beam'
    2 'beam'};

%tạo node
% Nodes=[NodID X  Y  Z]
CuaRao_Nodes;
L_span = 66.4; %m
% % Check the node coordinates as follows:
% % figure
% % plotnodes(Nodes,'numbering','on');
% % hold on

% Materials=[MatID E     n   u];
E_1	=	2.00E+11	;
E_2	=	2.00E+11	;
E_3	=	2.00E+11	;
E_4	=	2.00E+11	;
E_5	=	2.00E+11	;
E_6	=	2.00E+11	;
E_7	=	2.00E+11	;
E_8	=	2.00E+11	;
E_9	=	2.00E+11	;
E_10	=	2.00E+11	;

Materials=  [
1	E_1	0.3	7800
2	E_2	0.3	7800
3	E_3	0.3	7800
4	E_4	0.3	7800
5	E_5	0.3	7800
6	E_6	0.3	7800
7	E_7	0.3	7800
8	E_8	0.3	7800
9	E_9	0.3	7800
];   % steel

% Define Section and store at Sections variable
% Sections=[SecID A      ky   kz   Ixx Iyy Izz]

% Thanh xiên cổng cầu
A_1 = 0.041;
Iyy_1 = 0.001017492;
Izz_1 = 0.001032592;
% Thanh xiên hộp
A_2 = 0.0264;
Iyy_2 = 0.00048092;
Izz_2 = 0.00048092;

% Thanh xiên i
A_3 = 0.0202;
Ixx_3 = 0.000004;
Iyy_3 = 0.000431;
Izz_3 = 0.000143;

% Thanh biên trên
A_4 = 0.041862;
Ixx_4 = 0;
Iyy_4 = 0.000969;
Izz_4 = 0.001058;

% Thanh biên dưới
A_5 = 0.0304;
Ixx_5 = 0;
Iyy_5 = 0.001949653;
Izz_5 = 0.000733653;

% Dầm dọc dưới ray
A_6 = 0.0282;
Ixx_6 = 0;
Iyy_6 = 0.00832906;
Izz_6 = 0.00021374;

% Dầm ngang giữa
A_7 = 0.0292;
Ixx_7 = 0;
Iyy_7 = 0.007533493;
Izz_7 = 0.000143423;

% LK Dọc dưới
A_8 = 0.00026;
Ixx_8 = 0;
Iyy_8 = 3.38667E-06;
Izz_8 = 4.10167E-06;

% LK Dọc trên
A_9 = 0.006208;
Ixx_9 = 0;
Iyy_9 = 0.000108185;
Izz_9 = 1.60075E-05;

Sections=  [
1	A_1	Inf	Inf	0	Iyy_1	Izz_1
2	A_2	Inf	Inf	0	Iyy_2	Izz_2
3	A_3	Inf	Inf	0	Iyy_3	Izz_3
4	A_4	Inf	Inf	0	Iyy_4	Izz_4
5	A_5	Inf	Inf	0	Iyy_5	Izz_5
6	A_6	Inf	Inf	0	Iyy_6	Izz_6
7	A_7	Inf	Inf	0	Iyy_7	Izz_7
8	A_8	Inf	Inf	0	Iyy_8	Izz_8
9	A_9	Inf	Inf	0	Iyy_9	Izz_9
    ];

CuaRao_Elements;

% Plot elements in different colors in order to check the section definitions
% % leng_section = length(Sections);
% % colors = jet(leng_section);
% % % figure
% % for i = 1 : leng_section
% %     plotelem(Nodes,Elements(Elements(:,3)==Sections(i,1),:),Types,'Color',colors(i,:)...
% %         ,'Numbering','on','LineWidth',2);
% %     hold('on');
% % end
% % title('CuaRao bridge - Sections')

%dof
% Degrees of freedom
DOF=getdof(Elements,Types);

% Boundary conditions
seldof=[0.06
    [100;	300;]+0.01      % Gối cố định
    [100;	300;]+0.02
    [100;	300;]+0.03

    [108;	308;]+0.02      % Gối di động
    [108;	308;]+0.03];

DOF=removedof(DOF,seldof);

% Assembly of stiffness matrix K
[K,M]=asmkm(Nodes,Elements,Types,Sections,Materials,DOF);


% Eigenvalue problem
nMode=12;
[phi,omega]=eigfem(K,M,nMode);

f0 = omega/2/pi;
% % Display eigenfrequenties
% disp('Lowest eigenfrequencies [Hz]');
% disp(omega/2/pi);
% 
% % Plot eigenmodes
% for i = (1:8)
% figure;
% plotdisp(Nodes,Elements,Types,DOF,phi(:,i),'DispMax','off','LineWidth',2)
% title(['ModeShape ' num2str(i)])
% end
% 

