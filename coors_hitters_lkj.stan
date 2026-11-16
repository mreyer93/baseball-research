
data {
  int<lower=1> N;
  int<lower=1> S;
  array[N] int<lower=1, upper=S> season;
  array[N] int<lower=0> pa_h;
  array[N] int<lower=0> so_h;
  array[N] int<lower=0> pa_a;
  array[N] int<lower=0> so_a;
  array[N] int<lower=0> bip_h;
  array[N] int<lower=0> hb_h;
  array[N] int<lower=0> bip_a;
  array[N] int<lower=0> hb_a;
  int<lower=0, upper=1> use_home;
  int<lower=0, upper=1> gap_re;
  int<lower=0, upper=1> use_corr;
  real<lower=0> eta;
}
parameters {
  vector[S] mu_k;                  // road strikeout level by season (logit)
  vector[S] mu_b;                  // road BABIP level by season (logit)
  real gap_k;                      // average Coors strikeout gap (logit)
  real gap_b;                      // average Coors BABIP gap (logit)
  vector<lower=0>[4] tau;          // SDs of the four hitter effects
  cholesky_factor_corr[4] L;       // Cholesky factor of Omega
  matrix[4, N] z;                  // standard normal draws (non-centered)
}
transformed parameters {
  matrix[4, N] u;                  // rows: K talent, K gap, BABIP talent, BABIP gap
  if (use_corr == 1) {
    u = diag_pre_multiply(tau, L) * z;
  } else {
    u = diag_pre_multiply(tau, identity_matrix(4)) * z;
  }
}
model {
  mu_k ~ normal(-1.27, 0.5);       // logit(0.22)
  mu_b ~ normal(-0.85, 0.5);       // logit(0.30)
  gap_k ~ normal(0, 0.5);
  gap_b ~ normal(0, 0.5);
  tau ~ normal(0, 0.5);            // half-normal, because tau > 0
  L ~ lkj_corr_cholesky(eta);
  to_vector(z) ~ std_normal();
  so_a ~ binomial_logit(pa_a, mu_k[season] + u[1]');
  hb_a ~ binomial_logit(bip_a, mu_b[season] + u[3]');
  if (use_home == 1) {
    so_h ~ binomial_logit(pa_h, mu_k[season] + gap_k + gap_re * u[2]' + u[1]');
    hb_h ~ binomial_logit(bip_h, mu_b[season] + gap_b + gap_re * u[4]' + u[3]');
  }
}
generated quantities {
  corr_matrix[4] Omega = multiply_lower_tri_self_transpose(L);
  vector[N] k_road = inv_logit(mu_k[season] + u[1]');
  vector[N] b_road = inv_logit(mu_b[season] + u[3]');
}
