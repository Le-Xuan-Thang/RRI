% CHUONG DUONG BRIDGE - SPAN 10TH
% Finite element model - Static analysis
% Performer: PhD Student, MSc, B.Eng - Nguyen Ngoc Lan
% ================================================================
% The model used in the project 
% Title: Research on building a dynamic response monitoring system of railway 
% steel truss bridge structure under the influence of moving-load based on 
% artificial intelligence and vibration measurement results.
% Number code: T2022-CT-006TD
% Principle Investigator: Dr. Nguyen Xuan Tung
% Duration: 01/2022 - 12/2023
% -------------------------------------------------------------------------
% Units: m, kN, kNm, MPa
% Nodes=[NodID X Y Z]
clc; clear; close all;
L_span=89.28;
Nodes=Q_CDB_Nodes;
% -------------------------------------------------------------------------
% Element types -> {EltTypID EltName}
Types= {1 'beam'}; 
% -------------------------------------------------------------------------
% Sections=[SecID A ky kz Ixx Iyy Izz yt yb zt zb]
Sections=Q_CDB_Sections; 
% -------------------------------------------------------------------------
% Materials = [MatID E v density k];
EE=2.1e11;
v=0.3;
DEN=7850;
Materials=[1    EE  v DEN Inf];
% -------------------------------------------------------------------------
% Elements=[EltID TypID SecID MatID n1 n2 n3]
Elements=Q_CDB_Elements;
% Check node and element definitions as follows:
% figure
% plotnodes(Nodes,'Numbering','on','r');
% hold on
% plotelem(Nodes,Elements,Types,'Numbering','off','black','LineWidth',1);
% -------------------------------------------------------------------------
% Degrees of freedom
% Assemble a column matrix containing all DOFs at which stiffness is present in the model:
DOF=getdof(Elements,Types);
% DOF=dof_truss(NodeNum);
% Remove all DOFs equal to zero from the vector:
% 3D analysis
seldof=Q_CDB_DOF;
DOF=removedof(DOF,seldof);
% -------------------------------------------------------------------------
[K,M]=asmkm(Nodes,Elements,Types,Sections,Materials,DOF);
m= 290000;
m1= 8000;
M(2,2)=1.80+m;
M(3,3)=1.075+m;
M(4,4)=0.623+m;
M(5,5)= 0.646+m;
M(6,6)=0.636+m;
M(7,7)=0.958+m;
M(8,8)=1.424+m;
M(9,9)= 0.931+m;
M(10,10)=1.122+m;
M(11,11)=1.163+m;
M(22,22)=1.122+m;
M(23,23)=1.163+m;
M(24,24)=1.117+m;
M(25,25)= 2.70+m;
M(26,26)=3.91+m;
M(27,27)=1.90+m;
M(28,28)=0.61+m;
M(29,29)= 0.633+m;
M(30,30)=0.623+m;
M(31,31)=0.949+m;
M(42,42)=1.17+m;
M(43,43)=2.70+m;
M(44,44)=3.91+m;
M(45,45)= 1.9+m;
M(46,46)=0.61+m;
M(47,47)=0.633+m;
M(48,48)=0.623+m;
M(49,49)= 0.949+m;
M(50,50)=1.35+m;
M(51,51)=0.85+m;
M(202,202)= 0.443+m;
M(203,203)=0.188+m;
M(204,204)=0.651+m;
M(205,205)= 0.652+m;
M(206,206)=0.274+m;
M(207,207)=0.267+m;
M(208,208)=0.286+m;
M(209,209)= 0.114+m;
M(210,210)=0.384+m;
M(211,211)=0.379+m;
M(101,101)=0.193+m1;
M(102,102)= 0.595+m1;
M(103,103)=0.594+m1;
M(104,104)=0.418+m1;
M(105,105)= 0.429+m1;
M(106,106)=0.449+m1;
M(107,107)=0.193+m1;
M(108,108)=0.596+m1;
M(109,109)= 0.594+m1;
M(110,110)=0.418+m1;
M(301,301)=0.605+m1;
M(302,302)= 0.939+m1;
M(303,303)=1.33+m1;
M(304,304)=0.848+m1;
M(305,305)= 1.106+m1;
M(306,306)=1.147+m1;
M(307,307)=1.098+m1;
M(308,308)=2.69+m1;
M(309,309)= 3.89+m1;
M(310,310)=1.89+m1;
% -------------------------------------------------------------------------