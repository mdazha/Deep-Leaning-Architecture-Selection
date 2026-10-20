clear; clc; close all;

numDays = 7;
Tf = numDays * 1440;
tspan = (0:1:Tf)';

%% Meals (per day)
D1_base = 20; D2_base = 40; D3_base = 60;
tmeal1 = 0; tmeal2 = 4*60; tmeal3 = 9*60;

tmealvec = []; Dvec = [];
rng('default');
for d = 0:numDays-1
    D1 = D1_base * (0.7 + 0.6*rand());
    D2 = D2_base * (0.7 + 0.6*rand());
    D3 = D3_base * (0.7 + 0.6*rand());
    
    tmeal1 = d*1440 + 0  + randi([-50, 50]);
    tmeal2 = d*1440 + 4*60 + randi([-50, 50]);
    tmeal3 = d*1440 + 9*60 + randi([-50, 50]);
    
    daily_meals = sort([tmeal1, tmeal2, tmeal3]);
    tmealvec = [tmealvec; daily_meals'];
    Dvec     = [Dvec; D1*1000; D2*1000; D3*1000];
end

%% Type-1 Bergman + gut parameters
p4_T1D = 5/54;
berg = struct( ...
    'p1', 0.028735, 'p2', 0.028344, 'p3', 5.035e-5, 'p4', p4_T1D, ...
    'gamma', 0, 'h', 0, 'ib', 0, 'gb', 119.1858);

gut = struct( ...
    'k21', 0.0429, 'kmax', 0.0429, 'kmin', 0.0141, 'kabs', 0.2062, ...
    'b', 0.7612, 'c', 0.1372, 'f', 0.9, 'BW_n', 69.7098, 'Vg_n', 1.848);

x0 = [berg.gb; 0; berg.ib; 0; 0];
opts = odeset('AbsTol',1e-8,'RelTol',1e-8);

%% Basal + bolus insulin input
Ib_target = 7.3;
k_per_mg  = 1.2e-5;
tau_bolus = 1;
n_shape   = 2;
t_pre     = 10;

alpha_norm = exp(n_shape) / (n_shape^n_shape);

U_fun = @(t) basal_plus_bolus(t, tmealvec, Dvec, ...
                              p4_T1D*Ib_target, k_per_mg, ...
                              tau_bolus, n_shape, t_pre, alpha_norm);

[t, x] = ode45(@(tt,xx) model_type1_5(tt, xx, berg, U_fun, tmealvec, Dvec, gut), ...
               tspan, x0, opts);

G = x(:,1);
I = x(:,3);
U = arrayfun(U_fun, t);

dt = tspan(2) - tspan(1);
assert(all(mod(tmealvec, dt) == 0), 'Some meal times are off the 5-min grid.');

MealSize_mg  = zeros(length(tspan),1);
MealOnset    = zeros(length(tspan),1);

idx = int64(tmealvec./dt + 1);
MealSize_mg(idx) = MealSize_mg(idx) + Dvec;
MealOnset(idx)   = 1;

MealSize_g = MealSize_mg / 1000;

%% Plots 
figure('Color','w');

yyaxis left
plot(t, G, 'r', 'LineWidth', 4);
ylabel('Blood Glucose Concentration (mg/dL)');

yyaxis right
plot(t, U, 'b--', 'LineWidth', 2);
ylabel('Insulin (mU/L/min)', 'fontsize',36);

grid on;
xlabel('Time Index (minutes)','fontsize',36);
set(gca,'FontSize',36)
legend({'Blood Glucose (T1D)','Insulin'}, 'Location','best', 'fontsize',36)
xlim([t(1) t(end)])

ax = gca;
tmp = ax.YAxis(1).Color;
ax.YAxis(1).Color = ax.YAxis(2).Color;
ax.YAxis(2).Color = tmp;

tbl = table(t, G, U, MealOnset, MealSize_mg, 'VariableNames', ...
    {'Time','Glucose','Insulin','MealOnset','MealSize'});
writetable(tbl, 't1d_basal_bolus_meal_aligned.csv');
fprintf('Wrote t1d_basal_bolus_meal_aligned.csv  |  meals counted = %d\n', nnz(MealOnset));

function u = basal_plus_bolus(t, tmealvec, Dvec, basal, k_per_mg, tau_bolus, n, t_pre, alpha_norm)
    u = basal;
    for k = 1:numel(tmealvec)
        tk = tmealvec(k) - t_pre;
        s  = t - tk;
        if s > 0
            kern = ((s/tau_bolus).^n) .* exp(-s/tau_bolus) * alpha_norm;
            u = u + (k_per_mg * Dvec(k)) * kern;
        end
    end
end

function xdot = model_type1_5(t, x, berg, U_fun, tmealvec, Dvec, gut)
    G  = x(1); X = x(2); I = x(3); S = x(4); INT = x(5);

    p1 = berg.p1; p2 = berg.p2; p3 = berg.p3; p4 = berg.p4;
    ib = berg.ib; gb = berg.gb; gamma = berg.gamma; h = berg.h;

    k21 = gut.k21; kmax = gut.kmax; kmin = gut.kmin; c = gut.c; b = gut.b;
    kabs = gut.kabs; f = gut.f; BW_n = gut.BW_n; Vg_n = gut.Vg_n;

    qsto1 = 0;
    for ii = 1:length(tmealvec)
        if t >= tmealvec(ii)
            qsto1 = qsto1 + Dvec(ii) * exp(-k21*(t - tmealvec(ii)));
        end
    end
    qsto = qsto1 + S;

    D_total = sum(Dvec);
    alpha = 5 / (2*D_total*(1 - b));
    beta  = 5 / (2*D_total*c);
    kempty = kmin + (kmax - kmin)/2 * (tanh(alpha*(qsto - b*D_total)) ...
                                     - tanh(beta*(qsto - c*D_total)) + 2);

    rag = f * kabs * INT / (Vg_n * BW_n);

    u_exo = U_fun(t);

    u_basal_endog  = p4 * ib;
    u_gluc_stim    = max(0, gamma * (G - h));
    u_total = u_exo + u_basal_endog + u_gluc_stim;

    xdot = zeros(5,1);
    xdot(1) = -(X + p1)*G + p1*gb + rag;
    xdot(2) = -p2*X + p3*(I - ib);
    xdot(3) = -p4*I + u_total;
    xdot(4) = -kempty*S + k21*qsto1;
    xdot(5) = -kabs*INT + kempty*S;
end