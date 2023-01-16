#include <TMB.hpp>
// Spatial model of HIV transmission

// Vector logit
template<class Type>
vector<Type> logit(
  vector<Type> x
) {
  return log(x/(1-x));
}

// Vector inverse logit
template<class Type>
vector<Type> inv_logit(
  vector<Type> x
) {
  return 1/(1 + exp(-x));
}

// Matrix inverse logit
template<class Type>
matrix<Type> inv_logit(
  matrix<Type> x
) {
  return 1/(1 + exp(-x.array()));
}

using namespace Eigen;
using namespace density;

// Beta-binomial distribution
template<class Type>
Type dbbinom(Type k, Type n, Type a, Type b) {
  return lgamma(n+1) + lgamma(a+k) + lgamma(n-k+b) + lgamma(a+b) - (lgamma(k+1) + lgamma(n-k+1) + lgamma(n+a+b) + lgamma(a) + lgamma(b));
}

// Disease projection model
template<class Type>
matrix<Type> project_EPP(
  int n_region,
  int n_ts,
  vector<int> is_demog_ts,
  int n_stage,
  int n_nz,
  int n_age,
  int n_sex,
  Type dt,
  vector<Type> init_pop,
  matrix<Type> init_inf,
  matrix<Type> init_art,
  Eigen::SparseMatrix<Type> adjacency_m,
  matrix<Type> kappa_ts,
  matrix<Type> mort_stage,
  vector<int> from_mat_i,
  vector<int> to_mat_i,
  vector<int> from_region,
  vector<int> to_region,
  vector<int> from_i,
  vector<int> to_i,
  Type transm_reduction,
  matrix<int> to_m,
  vector<int> to_n,
  matrix<Type> entrant_m,
  matrix<Type> art_init_m,
  vector<Type> eta,
  vector<int> n_substage,
  vector<int> n_transition,
  int n_from,
  vector<int> from_group,
  vector<int> to_group,
  vector<int> from_substage,
  vector<int> stage_map,
  vector<int> from_compart,
  matrix<Type> transition_prob,
  matrix<Type> mort_bg,
  vector<int> substage_map,
  vector<Type> d_cd4,
  vector<int> from_age,
  vector<int> to_age,
  vector<Type> sex_IRR,
  matrix<Type> pop_m,
  vector<int> from_sex,
  matrix<Type> entrant_dist,
  matrix<Type> exit_m,
  matrix<Type> exit_dist
)
{
  // Containers for sparse matrix
  typedef Eigen::Triplet<Type> T;
  std::vector<T> tripletList;
  tripletList.reserve(n_nz);
  Eigen::SparseMatrix<Type> mat(n_sex * n_region * n_stage * n_age, n_sex * n_region * n_stage * n_age);
  // Filling in nonzero elements of sparse transition matrix
  for (size_t i = 0; i < n_nz; i++) {
    tripletList.push_back(T(to_i[i], from_i[i], Type(1.0)));
  }
  mat.setFromTriplets(tripletList.begin(), tripletList.end());

  // Final output matrix
  matrix<Type> out;
  out.setZero(n_sex * n_region * n_ts * n_age, n_stage + 4 + n_substage[1]);

  // Initializing storage
  matrix<Type> region_prev(n_region, n_substage[1] * n_sex);
  matrix<Type> region_no_art(n_region, n_substage[1] * n_sex);

  matrix<Type> region_numer;
  region_numer.setZero(n_region, n_substage[1] * n_sex);
  matrix<Type> region_denom;
  matrix<Type> region_denom_sqrt;
  region_denom.setZero(n_region, n_sex);
  region_denom_sqrt.setZero(n_region, n_sex);
  matrix<Type> region_art;
  region_art.setZero(n_region, n_substage[1] * n_sex);
  vector<Type> region_aids_mort(n_region * n_age * n_sex);
  vector<Type> weighted_prev(n_region);
  weighted_prev.setZero(n_region);

  vector<Type> region_age_sex_pop;
  region_age_sex_pop.setZero(n_region * n_age * n_sex);

  vector<Type> art_adj_prev_age(n_region * n_age * n_sex);

  matrix<Type> cumul_sum;
  cumul_sum.setZero(n_region, n_sex);

  vector<Type> tmp_cumul_sum;
  matrix<Type> incidence_region;
  incidence_region.setZero(n_region, n_sex);
  vector<Type> incidence_region_age;
  incidence_region_age.setZero(n_region * n_age * n_sex);

  vector<Type> tmp_kappa;
  tmp_kappa.setZero(n_region);

  matrix<Type> art_adj_prev;
  art_adj_prev.setZero(n_region, n_substage[1] * n_sex);

  Type insert_val;
  vector<Type> tmp_segment(n_stage);
  vector<Type> art_change(n_sex * n_region * n_substage[2] * n_age);
  Type deriv_sum;
  Type adjusted_deriv_sum;
  vector<Type> adjusted_derivs(to_m.cols());

  vector<Type> tmp_derivs(to_m.cols());

  vector<Type> curr_entrants(n_region);
  vector<Type> curr_exits(n_region);
  vector<Type> allocated_entrants;

  // Setting current state
  vector<Type> curr_state;
  vector<Type> tmp_init_prev(n_substage[1]);
  vector<Type> tmp_init_art(n_substage[2]);
  Type tmp_init_susecpt = 0;
  vector<Type> tmp_pop(n_age * n_sex);

  int curr_from_mat_i;
  int curr_to_mat_i;
  int curr_from_group;
  int curr_to_group;
  int curr_from_age;
  int curr_to_age;
  int curr_from_sex;

  matrix<Type> tmp_block_inf(n_age, n_substage[1]);
  matrix<Type> tmp_block_art(n_age, n_substage[2]);

  vector<Type> age_prev(n_age);
  vector<Type> age_art(n_age);
  vector<Type> dumb_test(n_substage[1]);
  curr_state.setZero(n_region * n_stage * n_age * n_sex);

  // Fill in initial state
  for (size_t r = 0; r < n_region; r++) {
    tmp_init_prev.setZero(n_substage[1]);
    tmp_init_art.setZero(n_substage[2]);
    tmp_init_susecpt = 0;
    for (size_t g = 0; g < n_sex; g++) {
      tmp_block_inf = init_inf.block(g * n_age * n_region + r * n_age, 0, n_age, n_substage[1]);
      tmp_block_art = init_art.block(g * n_age * n_region + r * n_age, 0, n_age, n_substage[2]);

      tmp_init_prev = tmp_block_inf.colwise().sum().array();
      tmp_init_art = tmp_block_art.colwise().sum().array();
      tmp_pop = init_pop[g * n_region * n_age + r * n_age];
      region_denom(r, g) += tmp_pop.sum();

      region_numer.block(r, g * n_substage[1], 1, n_substage[1]) = tmp_init_prev.transpose();
      region_art.block(r, g * n_substage[2], 1, n_substage[2]) = tmp_init_art.transpose();

      age_prev = tmp_block_inf.rowwise().sum();

      for (size_t a = 0; a < n_age; a++) {
        curr_state(g * n_region * n_age * n_stage + r * n_stage * n_age + a * n_stage) = tmp_pop[a] - age_prev[a];
        tmp_init_prev = tmp_block_inf.row(a);
        tmp_init_art = tmp_block_art.row(a);
        curr_state.segment(g * n_region * n_age * n_stage + r * n_stage * n_age + a * n_stage + n_substage[0], n_substage[1]) = tmp_block_inf.row(a) - tmp_block_art.row(a);
        curr_state.segment(g * n_region * n_age * n_stage + r * n_stage * n_age + a * n_stage + n_substage[0] + n_substage[1], n_substage[2]) = tmp_block_art.row(a);
      }
      region_denom_sqrt(r, g) = pow(region_denom(r, g), 0.5);
    }
  }
  for (size_t g = 0; g < n_sex; g++) {
    for (size_t s = 0; s < n_substage[1]; s++) {
      region_prev.col(g * n_substage[1] + s) = region_numer.col(g * n_substage[1] + s).array() / region_denom.col(g).array();
    }
  }
  region_no_art = region_numer - region_art;

  int demog_year_count = 0;
  vector<Type> replacement_state;
  Type deriv_val;
  Type diag_val;
  vector<Type> pop_ratio(n_region * n_age * n_sex);
  vector<Type> tmp_region_pop;
  vector<Type> tmp_region_prev;
  vector<Type> tmp_region_odds;
  matrix<Type> art_init_count(n_region * n_age * n_sex, n_substage[1]);
  vector<Type> tmp_segment_stage(n_stage);
  vector<Type> tmp_exit_dist(n_stage);
  vector<Type> adj_odds_exits(n_stage);
  vector<Type> adj_prev_exits;
  vector<Type> adj_exits(n_stage);
  vector<Type> tmp_entrant_dist(n_stage);
  vector<Type> adj_odds_entrants(n_stage);
  vector<Type> adj_prev_entrants;
  vector<Type> adj_entrants(n_stage);
  // Main integration loop
  for (size_t t = 0; t < n_ts; t++) {
    art_init_count.setZero(n_region * n_age * n_sex, n_substage[1]);

    // Adding exogenous net entrants to S
    replacement_state.setZero(curr_state.size());
    for (size_t g = 0; g < n_sex; g++) {
      curr_entrants = entrant_m.block(g * n_region, t, n_region, 1);
      curr_exits = exit_m.block(g * n_region, t, n_region, 1);
      tmp_exit_dist = exit_dist.row(g * n_ts + t).array();
      tmp_entrant_dist = entrant_dist.row(g * n_ts + t).array();
      for (size_t r = 0; r < n_region; r++) {
          tmp_region_pop = curr_state.segment(g * n_region * n_stage * n_age + r * n_stage * n_age, n_stage);
          tmp_region_prev = tmp_region_pop / sum(tmp_region_pop);
          tmp_region_odds = tmp_region_prev/tmp_region_prev[0];

          adj_odds_exits = tmp_exit_dist * tmp_region_odds;
          adj_prev_exits = adj_odds_exits / sum(adj_odds_exits);
          adj_exits = adj_prev_exits * exit_m(g * n_region + r, t);

          adj_odds_entrants = tmp_entrant_dist * tmp_region_odds;
          adj_prev_entrants = adj_odds_entrants / sum(adj_odds_entrants);
          adj_entrants = adj_prev_entrants * entrant_m(g * n_region + r, t);

          curr_state.segment(g * n_region * n_stage * n_age + r * n_stage * n_age, n_stage) += adj_entrants.array() - adj_exits.array();
      }
    }
    for (size_t i = 0; i < curr_state.size(); i++) {
      if (curr_state[i] <= 0.0) {
        curr_state[i] = 0.0;
      }
    }

    tmp_kappa = kappa_ts.row(t);
    art_change = art_init_m.row(t);
    cumul_sum.setZero(n_region, n_sex);

    // Finding weighted ART-adjusted prevalence for incidence
    art_adj_prev = region_no_art.array() + (1 - transm_reduction) * region_art.array();

    for (size_t g = 0; g < n_sex; g++) {
      tmp_cumul_sum.setZero(n_region);
      vector<Type> tmp1 = region_denom_sqrt.col(g).array();
      vector<Type> tmp3 = region_denom_sqrt.col(-1 * (g-1)).array();

      for (size_t s = 0; s < n_substage[1]; s++) {
        vector<Type> tmp2 = art_adj_prev.col(g * n_substage[1] + s).array();
        vector<Type> tmp_adj_prev = tmp2/tmp1;
        weighted_prev = adjacency_m * tmp_adj_prev;
        // Calculating incidence probability
        // Switching men and women so that incidence depends on
        // opposite-sex prevalence. Not 100% accurate but very close
        tmp_cumul_sum += weighted_prev * tmp_kappa * d_cd4[s] * exp(sex_IRR[t] * g);
      }
      cumul_sum.col(-1 * (g - 1)) = tmp_cumul_sum / tmp3;
    }
    incidence_region = -log(1 - exp(cumul_sum.array()))/dt;
    Type max_exits;
    for (size_t g = 0; g < n_sex; g++) {
      for (size_t a = 0; a < n_age; a++) {
        max_exits = (mort_bg(g * n_age + a, t) + mort_stage(0 * n_age * n_sex + g * n_age + a, t));
        for (size_t r = 0; r < n_region; r++) {
          incidence_region_age[g * n_age * n_region + r * n_age + a] = (1 - max_exits) * (1-exp(-dt * cumul_sum(r, g)));
        }
      }
    }

    region_aids_mort.setZero(n_region * n_age * n_sex);
    // Adding incidence probability and ART initiation probability to
    // transition rate storage
    for (size_t i = 0; i < n_from; i++) {
      deriv_sum = 0;
      tmp_derivs.setZero(to_m.cols());
      curr_from_mat_i = from_mat_i[i];
      curr_from_group = from_group[i];
      curr_from_age = from_age[i];
      curr_from_sex = from_sex[i];

      diag_val = 1 - (mort_stage(from_compart[i] * n_age * n_sex + curr_from_sex * n_age + curr_from_age, t) + mort_bg(curr_from_sex * n_age + curr_from_age, t));
      for (size_t j = 0; j < to_n[i]; j++) {
        curr_to_mat_i = to_m(i,j);
        curr_to_group = to_group[to_m(i,j)];
        curr_to_age = to_age[to_m(i,j)];

        if (curr_from_mat_i == to_m(i,j)) {
        } else {
          if (curr_from_group == 0) {
            diag_val -= mat.coeffRef(to_m(i,j), i) = incidence_region_age[curr_from_sex * n_region * n_age + from_region[i] * n_age + curr_from_age];
          } else if (curr_from_group == 1) {
            if (curr_to_group == 1) {
                diag_val -= mat.coeffRef(to_m(i,j), i) = transition_prob(from_substage[i] * n_sex * n_age + curr_from_sex * n_age + curr_from_age, t);
            } else if (curr_to_group == 2) {
                diag_val -= mat.coeffRef(to_m(i,j), i) = art_change[curr_from_sex * n_substage[1] * n_region * n_age + from_substage[i] * n_region * n_age + from_region[i] * n_age + curr_from_age];
                art_init_count(curr_from_sex * n_region * n_age + from_region[i] * n_age + curr_from_age, from_substage[i]) += art_change[curr_from_sex * n_substage[1] * n_region * n_age + from_substage[i] * n_region * n_age + from_region[i] * n_age + curr_from_age] * curr_state[i];
            }
          } else if (curr_from_group == 2) {
            diag_val -= mat.coeffRef(to_m(i,j), i) = eta[0];
          }
        }
      }
      mat.coeffRef(i, i) = diag_val;
    }

    // Find new state based on previous state
    curr_state = mat * curr_state;
    region_age_sex_pop.setZero(n_sex * n_region * n_age);

    for (size_t g = 0; g < n_sex; g++) {
      for (size_t r = 0; r < n_region; r++) {
        for (size_t a = 0; a < n_age; a++) {
          tmp_segment_stage = curr_state.segment(g * n_region * n_stage * n_age + r * n_stage * n_age + a * n_stage, n_stage);
          curr_state.segment(g * n_region * n_stage * n_age + r * n_stage * n_age + a * n_stage, n_stage) *= pop_m(g * n_region * n_age + r * n_age + a, t) / tmp_segment_stage.sum();
        }
      }
    }
    // Intialize storage for aggregates
    region_numer.setZero(n_region, n_substage[1] * n_sex);
    region_denom.setZero(n_region, n_sex);
    region_denom_sqrt.setZero(n_region, n_sex);
    region_art.setZero(n_region, n_substage[1] * n_sex);

    // Find aggregates for incidence and ART coverage calculation
    // at beginning of next step
    for (size_t r = 0; r < n_region; r++) {
      for (size_t g = 0; g < n_sex; g++) {
        // Restrict to a single region
        tmp_segment = curr_state.segment(g * n_region * n_stage * n_age + r * n_stage * n_age, n_stage * n_age);
        for (size_t a = 0; a < n_age; a++) {
          for (size_t s = 0; s < n_stage; s++) {
            if (s > 0) {
              // All PLHIV
              region_numer(r, g * n_substage[1] + substage_map[s]) += tmp_segment[a * n_stage + s];

              // PLHIV with ART
              if (stage_map[s] == 2) {
                region_art(r, g * n_substage[1] + substage_map[s]) += tmp_segment[a * n_stage + s];
              }
            }
            // Everyone
            region_denom(r, g) += tmp_segment[a * n_stage + s];
            region_age_sex_pop[g * n_region * n_age + r * n_age + a] += tmp_segment[a * n_stage + s];
          }
          // Fill output matrix for S, I, and A
          for (size_t s = 0; s < n_stage; s++) {
            out(t * n_region * n_age * n_sex + g * n_region * n_age + r * n_age + a, s) = tmp_segment[a * n_stage + s];
          }
        }
        region_denom_sqrt(r, g) = pow(region_denom(r, g), 0.5);
      }
    }
    region_prev.setZero(n_region, n_substage[1] * n_sex);

    // Fill output for incidence probabilities and rates
    for (size_t g = 0; g < n_sex; g++) {
      for (size_t s = 0; s < n_substage[1]; s++) {
        region_prev.col(g * n_substage[1] + s) = region_numer.col(g * n_substage[1] + s).array() / region_denom.col(g).array();
      }
      out.block(t * n_region * n_age * n_sex + g * n_region, n_stage + 1, n_region, 1) = cumul_sum.col(g);
    }
    region_no_art = region_numer - region_art;
    out.block(t * n_region * n_age * n_sex, n_stage, n_region * n_age * n_sex, 1) = incidence_region_age;
    if (t > 0) {
      out.block(t * n_region * n_age * n_sex, n_stage + 2, n_region * n_age * n_sex, 1) = out.block((t-1) * n_region * n_age * n_sex, 0, n_region * n_age * n_sex, 1).array() * out.block(t * n_region * n_age * n_sex, n_stage, n_region * n_age * n_sex, 1).array();
    }
    out.block(t * n_region * n_age * n_sex, n_stage + 3, n_region * n_age * n_sex, 1) = region_aids_mort;
    out.block(t * n_region * n_age * n_sex, n_stage + 4, n_region * n_age * n_sex, n_substage[1]) = art_init_count;

  } // End integration

  // Return full projection
  return out;
}

// Objective function (TMB requirement)
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Load data from R
  DATA_INTEGER(n_region); // No. regions
  DATA_INTEGER(n_ts); // No. time steps
  DATA_IVECTOR(is_demog_ts); // Time points for demography to occur
  DATA_INTEGER(n_stage); // No. disease stages
  DATA_INTEGER(n_age); // No. age groups (one)
  DATA_INTEGER(n_sex); // No. of sex values/subpopulations
  DATA_SCALAR(dt); // Time step

  DATA_VECTOR(init_pop); // Initial population
  DATA_VECTOR(init_pop_share); // Initial spatial distribution
  DATA_SCALAR(init_prev); // Initial national prevalence

  DATA_INTEGER(n_nz); // No. nonzero transitions
  DATA_IVECTOR(n_transition); // Not used
  DATA_INTEGER(n_from); // No. sending compartments
  DATA_IVECTOR(from_i); // ID of sending comparmtents
  DATA_IVECTOR(to_i); // ID of receiving compartments
  DATA_SPARSE_MATRIX(spline_ts); // Splines for transmission rate
  DATA_SPARSE_MATRIX(spline_ts_irr); // Splines for transmission rate sex ratio
  DATA_SPARSE_MATRIX(spline_ts_art_init_sex); // Splines ART initiation sex ratio
  DATA_MATRIX(mort_stage); // Stage-specific mortality
  DATA_MATRIX(mort_bg); // Background mortality
  DATA_SPARSE_MATRIX(adjacency_m); // Adjacency of districts
  DATA_IVECTOR(from_mat_i); // From matrix index
  DATA_IVECTOR(to_mat_i); // To matrix index
  DATA_IVECTOR(from_region); // From region
  DATA_IVECTOR(to_region); // To region

  DATA_IVECTOR(from_group); // Broad compartment
  DATA_IVECTOR(to_group);

  DATA_IVECTOR(from_substage); // Granular compartment
  DATA_IVECTOR(from_sex);

  DATA_INTEGER(is_simulation); // Don't evaluate likelihood?

  // Survey prevalence data
  DATA_INTEGER(n_in_survey);
  DATA_INTEGER(n_survey);
  DATA_IVECTOR(has_survey_i);
  DATA_IVECTOR(survey_i);
  DATA_VECTOR(prev_tested);
  DATA_VECTOR(prev_pos);
  DATA_IVECTOR(prev_in_sample);
  DATA_SCALAR(transm_reduction);

  DATA_VECTOR(art_pos);
  DATA_IVECTOR(survey_art_i);
  DATA_VECTOR(recent_pos);
  DATA_SCALAR(MDRI);

  // ANC data
  DATA_IVECTOR(anc_i);
  DATA_VECTOR(anc_tested);
  DATA_VECTOR(anc_pos);
  DATA_INTEGER(any_recent);
  DATA_IVECTOR(n_prev_survey_contrib);
  DATA_IMATRIX(prev_obs_m);


  // Survey recency data
  DATA_INTEGER(n_in_recent);
  DATA_INTEGER(n_recent);
  DATA_IVECTOR(has_recent_i);
  DATA_IVECTOR(n_recent_survey_contrib);
  DATA_IMATRIX(recent_obs_m);
  DATA_VECTOR(recent_tested);
  DATA_IVECTOR(recent_in_sample);

  // Survey ART data
  DATA_INTEGER(any_art_test);
  DATA_INTEGER(n_in_art_test);
  DATA_INTEGER(n_art_test);
  DATA_IVECTOR(has_art_test_i);
  DATA_IVECTOR(n_art_test_survey_contrib);
  DATA_IMATRIX(art_test_obs_m);
  DATA_VECTOR(art_tested);
  DATA_IVECTOR(art_test_in_sample);

  // ANC data alignment
  DATA_INTEGER(n_in_anc);
  DATA_INTEGER(n_anc);
  DATA_IVECTOR(has_anc_i);
  DATA_IVECTOR(n_anc_survey_contrib);
  DATA_IMATRIX(anc_obs_m);
  DATA_IVECTOR(site_i);
  DATA_IVECTOR(anc_in_sample);
  DATA_VECTOR(anc_ts);

  // ART programme data
  DATA_VECTOR(art_count);
  DATA_IVECTOR(n_art_count_survey_contrib);
  DATA_IMATRIX(art_count_obs_m);
  DATA_INTEGER(n_in_art_count);
  DATA_INTEGER(n_art_count);
  DATA_IVECTOR(art_count_in_sample);
  // DATA_IVECTOR(has_art_count_i);
  DATA_IVECTOR(art_count_region_i);

  DATA_IMATRIX(to_m); // To indices
  DATA_IVECTOR(to_n); // No. of receiving compartments
  DATA_INTEGER(any_anc); // Any ANC data?

  // Entrants and exits
  DATA_MATRIX(entrant_m);
  DATA_MATRIX(entrant_dist);
  DATA_MATRIX(exit_m);
  DATA_MATRIX(exit_dist);

  DATA_SPARSE_MATRIX(art_init_design); // Splines for ART initiation
  DATA_MATRIX(art_design_binary); // Is subgroup eligible?

  DATA_INTEGER(post_scaleup); // Is there any ART?

  DATA_INTEGER(art_ll); // Which ART likelihood to use

  DATA_MATRIX(dist_m); // Not used
  DATA_IVECTOR(art_count_ts_i); // Time index for ART data

  DATA_IMATRIX(wide_nb_m); // Wide neighbor list
  DATA_IVECTOR(n_nb); // No. of neighbors
  DATA_IVECTOR(n_substage); // No. of substages by compartment
  DATA_MATRIX(init_substage_dist); // Initial distribution by substage
  // Utilities
  DATA_IVECTOR(stage_map);
  DATA_IVECTOR(substage_map);
  DATA_IVECTOR(from_compart);

  DATA_MATRIX(transition_prob); // Movement between CD4 categories
  DATA_MATRIX(cd4_elig); // Treatment eligibility

  DATA_IVECTOR(from_age); // From age index
  DATA_IVECTOR(to_age); // To age index
  DATA_SPARSE_MATRIX(spline_m_attract); // Splines for ART attendance model

  DATA_MATRIX(mort_ratio_m); // Mortality ratio for ART initiation

  DATA_MATRIX(pop_m); // Population
  DATA_SCALAR(tol_val); // Minimum for positive values

  DATA_SCALAR(exp_mean); // Mean self-attraction for ART attendance
  DATA_VECTOR(kappa_times); // Time series for transmission rates
  DATA_VECTOR(kappa_w); // Weights for ARIMA component (one)

  DATA_SCALAR(delta_t);

  DATA_VECTOR(art_times); // Linear time series for ART

  DATA_INTEGER(diff_ord); // Order of differencing for kappa
  DATA_INTEGER(diff_ord_art); // Order of differencing for alpha
  DATA_INTEGER(use_autoregressive); // Use autoregressive model for kappa
  DATA_INTEGER(model_innovations); // Model innnovations for kappa
  DATA_INTEGER(model_innovations_art); // Model innovations for alpha
  DATA_INTEGER(anc_likelihood); // Which ANC likelihood to use
  DATA_IVECTOR(anc_agg_i); // Is ANC a "census" observation?

  // Read in paramters from R
  PARAMETER_VECTOR(init_logit_prev_i); // Initial regional prevalence
  PARAMETER(init_logit_prev_mean); // Mean preavlence
  PARAMETER(log_sigma_prev_i);

  PARAMETER(logit_anc_delta); // ANC offset
  PARAMETER_VECTOR(logit_anc_delta_site); // Site-specific offset
  PARAMETER(log_sigma_anc_delta_site); // Site offset SD
  PARAMETER(logit_anc_delta_b_t); // ANC bias slope
  PARAMETER_VECTOR(logit_anc_delta_b_t_i); // Site-specific bias slope
  PARAMETER(log_sigma_anc_delta_b_t_i); // Site slope Sd

  PARAMETER(art_init_intercept); // Mean alpha
  PARAMETER_VECTOR(art_init_rw); // Shared alpha AR
  PARAMETER(log_sigma_art_init_rw); // Shared alpha AR scale
  PARAMETER(logit_phi_art_init_rw); // Shared alpha AR correlation

  PARAMETER_VECTOR(art_init_b_r); // alpha region intercept
  PARAMETER(log_sigma_art_init_b_r);

  PARAMETER(art_init_b_t); // alpha slope
  PARAMETER_VECTOR(art_init_b_t_i); // alpha region slope
  PARAMETER(log_sigma_art_init_b_t_i);

  PARAMETER_MATRIX(art_init_rw_i); // alpha region AR
  PARAMETER_MATRIX(eps_art_init_rw_i);
  PARAMETER(log_sigma_art_init_rw_i);
  PARAMETER(logit_phi_art_init_rw_i);

  PARAMETER_VECTOR(art_init_b_s_i); // alpha sex-region intercept
  PARAMETER(log_sigma_art_init_b_s_i);

  PARAMETER(art_init_b_s); // alpha sex effects
  PARAMETER_VECTOR(art_init_b_s_t); // alpha sex-time slope
  PARAMETER(log_sigma_art_init_b_s_t); // alpha sex-time slope SD
  PARAMETER_MATRIX(art_init_b_s_t_i); // alpha sex-time-space slope
  PARAMETER(log_sigma_art_init_b_s_t_i); // alph sex-time-space slope SD
  PARAMETER(art_init_b_mort); // alpha mortality effect (not used)
  PARAMETER(log_art_init_b_elig); // alpha eligibility effect (not used)
  PARAMETER(art_init_b_elig_s); // alpha eligibility sex interaction (not used)

  PARAMETER_VECTOR(logit_initial_art_i); // Not used
  PARAMETER(logit_initial_art_mean); // Not used
  PARAMETER(log_sigma_initial_art_i); // Not used
  PARAMETER(log_overdispersion); // Programme data hyperpars
  PARAMETER(log_omega); // Programme data hyperpars

  PARAMETER(log_eta); // ART LTFU

  PARAMETER_VECTOR(log_ART_OR); // ART attendance intercept
  PARAMETER_VECTOR(log_ART_attract); // ART attendance attractiveness
  PARAMETER(logit_ART_attract_mean);
  PARAMETER(log_sigma_ART_attract);

  PARAMETER_VECTOR(log_ART_repel); // ART attendance repellence
  PARAMETER(log_ART_repel_mean);
  PARAMETER(log_sigma_ART_repel);
  PARAMETER(beta_dist);

  PARAMETER_VECTOR(log_d_cd4); // Not used

  PARAMETER(init_prev_f); // Initial prevalence sex effect
  PARAMETER(init_art_f); // Not used
  PARAMETER_VECTOR(init_prev_f_i); // Initial prevalence sex-region
  PARAMETER_VECTOR(init_art_f_i); // Not used
  PARAMETER(log_sigma_init_prev_f);
  PARAMETER(log_sigma_init_art_f);

  PARAMETER_VECTOR(log_sex_IRR); // Sex IRR
  PARAMETER_VECTOR(b_time); // ART attendance slope
  PARAMETER_MATRIX(b_time_i); // ART attendance region slope
  PARAMETER(log_sigma_b_time_i);
  PARAMETER(log_spline_smooth_attract);

  PARAMETER(kappa_r_0); // Not used
  PARAMETER_VECTOR(kappa_r_0_i); //
  PARAMETER(log_sigma_kappa_r_0);

  PARAMETER(log_kappa_r_inf_delta); // Not used
  PARAMETER_VECTOR(log_kappa_r_inf_delta_i);
  PARAMETER(log_sigma_kappa_r_inf_delta);

  PARAMETER(kappa_t_mid); // Not used
  PARAMETER_VECTOR(kappa_t_mid_i);
  PARAMETER(log_sigma_kappa_t_mid);

  PARAMETER(log_kappa_alpha); // Not used
  PARAMETER_VECTOR(log_kappa_alpha_i);
  PARAMETER(log_sigma_kappa_alpha);

  PARAMETER_VECTOR(kappa_rw); // Kappa ARIMA component
  PARAMETER_VECTOR(eps_kappa_rw);
  PARAMETER(log_sigma_kappa_rw);
  PARAMETER(logit_phi_kappa_rw);

  PARAMETER_MATRIX(kappa_rw_i); // Kappa ARIMA-region
  PARAMETER_MATRIX(eps_kappa_rw_i);
  PARAMETER(log_sigma_kappa_rw_i);
  PARAMETER(logit_phi_kappa_rw_i);

  PARAMETER_VECTOR(logit_anc_corr); // ANC beta-binomial correlation

  // negative log-posterior density
  Type nll = 0;

  // Inverse link for hyperpars
  Type sigma_kappa_rw = exp(log_sigma_kappa_rw);
  Type phi_kappa_rw = 2 / (1 + exp(-logit_phi_kappa_rw)) - 1;

  Type sigma_kappa_rw_i = exp(log_sigma_kappa_rw_i);
  // Type phi_kappa_rw_i = logit_phi_kappa_rw_i;
  Type phi_kappa_rw_i = 1 / (1 + exp(-logit_phi_kappa_rw_i));
  if (use_autoregressive == 0) phi_kappa_rw_i = 0;

  Type sigma_art_init_rw = exp(log_sigma_art_init_rw);
  Type phi_art_init_rw = 2 / (1 + exp(-logit_phi_art_init_rw)) - 1;

  Type sigma_art_init_rw_i = exp(log_sigma_art_init_rw_i);
  Type phi_art_init_rw_i = 2 / (1 + exp(-logit_phi_art_init_rw_i)) - 1;

  // Handling ARIMA innovations vs actual values
  vector<Type> kappa_rw_coef;
  if (model_innovations == 0) {
    kappa_rw_coef = kappa_rw;
  }
  else if (model_innovations == 1){
    kappa_rw_coef.setZero(eps_kappa_rw.size()+1);
    kappa_rw_coef(1) = eps_kappa_rw(0);
    for (size_t i = 2; i < kappa_rw_coef.size(); i++) {
      if (diff_ord_art == 0) {
        kappa_rw_coef(i) = phi_kappa_rw * kappa_rw_coef(i-1) + eps_kappa_rw(i-1);
      }
      if (diff_ord_art == 1) {
        kappa_rw_coef(i) = kappa_rw_coef(i-1) + phi_kappa_rw * (kappa_rw_coef(i-1)-kappa_rw_coef(i-2)) + eps_kappa_rw(i-1);
      }
      if (diff_ord_art == 2) {
        if (i == 2) {
          kappa_rw_coef(i) = 2 * kappa_rw_coef(i-1) - kappa_rw_coef(i-2) + eps_kappa_rw(i-1);
        } else {
          kappa_rw_coef(i) = 2 * kappa_rw_coef(i-1) - kappa_rw_coef(i-2) + phi_kappa_rw * ((kappa_rw_coef(i-1)-kappa_rw_coef(i-2)) - (kappa_rw_coef(i-2) - kappa_rw_coef(i-3))) + eps_kappa_rw(i-1);
        }
      }
    }
  }

  // Calculate kappa ARIMA component coefficients
  matrix<Type> kappa_rw_i_coef;
  if (model_innovations == 0) {
    kappa_rw_i_coef = kappa_rw_i;
  }
  else if (model_innovations == 1){
    kappa_rw_i_coef.setZero(eps_kappa_rw_i.rows()+1, n_region);
    kappa_rw_i_coef.row(1) = eps_kappa_rw_i.row(0);
    for (size_t i = 2; i < kappa_rw_i_coef.rows(); i++) {
      if (diff_ord_art == 0) {
        kappa_rw_i_coef.row(i) = phi_kappa_rw_i * kappa_rw_i_coef.row(i-1) + eps_kappa_rw_i.row(i-1);
      }
      if (diff_ord_art == 1) {
        kappa_rw_i_coef.row(i) = kappa_rw_i_coef.row(i-1) + phi_kappa_rw_i * (kappa_rw_i_coef.row(i-1)-kappa_rw_i_coef.row(i-2)) + eps_kappa_rw_i.row(i-1);
      }
      if (diff_ord_art == 2) {
        if (i == 2) {
          kappa_rw_i_coef.row(i) = 2 * kappa_rw_i_coef.row(i-1) - kappa_rw_i_coef.row(i-2) + eps_kappa_rw_i.row(i-1);
        } else {
          kappa_rw_i_coef.row(i) = 2 * kappa_rw_i_coef.row(i-1) - kappa_rw_i_coef.row(i-2) + phi_kappa_rw_i * ((kappa_rw_i_coef.row(i-1)-kappa_rw_i_coef.row(i-2)) - (kappa_rw_i_coef.row(i-2) - kappa_rw_i_coef.row(i-3))) + eps_kappa_rw_i.row(i-1);
        }
      }
    }
  }

  // Calculate alpha ARIMA space time coefficients
  matrix<Type> art_init_rw_i_coef;
  if (model_innovations_art == 0) {
    art_init_rw_i_coef = art_init_rw_i;
  }
  else if (model_innovations_art == 1){
    art_init_rw_i_coef.setZero(eps_art_init_rw_i.rows()+1, n_region);
    art_init_rw_i_coef.row(1) = eps_art_init_rw_i.row(0);
    for (size_t i = 2; i < art_init_rw_i_coef.rows(); i++) {
      if (diff_ord == 0) {
        art_init_rw_i_coef.row(i) = phi_art_init_rw_i * art_init_rw_i_coef.row(i-1) + eps_art_init_rw_i.row(i-1);
      }
      if (diff_ord == 1) {
        art_init_rw_i_coef.row(i) = art_init_rw_i_coef.row(i-1) + phi_art_init_rw_i * (art_init_rw_i_coef.row(i-1)-art_init_rw_i_coef.row(i-2)) + eps_art_init_rw_i.row(i-1);
      }
      if (diff_ord == 2) {
        if (i == 2) {
          art_init_rw_i_coef.row(i) = 2 * art_init_rw_i_coef.row(i-1) - art_init_rw_i_coef.row(i-2) + eps_art_init_rw_i.row(i-1);
        } else {
          art_init_rw_i_coef.row(i) = 2 * art_init_rw_i_coef.row(i-1) - art_init_rw_i_coef.row(i-2) + phi_art_init_rw_i * ((art_init_rw_i_coef.row(i-1)-art_init_rw_i_coef.row(i-2)) - (art_init_rw_i_coef.row(i-2) - art_init_rw_i_coef.row(i-3))) + eps_art_init_rw_i.row(i-1);
        }
      }
    }
  }

  // Log-prior and log-likelihood
  Type lprior = 0;
  Type llikelihood = 0;

  // Not used
  vector<Type> kappa_alpha = (log_kappa_alpha_i + log_kappa_alpha);
  vector<Type> kappa_r_inf_delta = -exp(log_kappa_r_inf_delta_i + log_kappa_r_inf_delta);
  vector<Type> kappa_r_0_v = kappa_r_0_i + kappa_r_0;
  vector<Type> kappa_t_mid_v = kappa_t_mid_i + kappa_t_mid;

  // Fixed LTFU rate
  Type exp_eta = exp(log_eta);
  Type eta_scalar = 1 - exp(-dt * exp_eta);
  vector<Type> eta(n_region*n_substage[2]);
  for (size_t i = 0; i < eta.size(); i++) {
    eta[i] = eta_scalar;
  }

  // Fixed CD4 infectiousness ratios
  vector<Type> d_cd4;
  d_cd4.setZero(n_substage[1]);
  for (size_t s = 1; s < n_substage[1]; s++) {
    d_cd4[s] = exp(log_d_cd4[s-1]);
  }

  // Predict sex IRR
  vector<Type> sex_IRR = spline_ts_irr * log_sex_IRR;
  REPORT(sex_IRR);

  // Predict initial prevalence and ART
  matrix<Type> init_art;
  init_art.setZero(n_region * n_age * n_sex, n_substage[2]);
  matrix<Type> init_inf_no_ART;
  init_inf_no_ART.setZero(n_region * n_age * n_sex, n_substage[1]);
  for (size_t s = 0; s < n_sex; s++) {
    for (size_t r = 0; r < n_region; r++) {
      for (size_t a = 0; a < n_age; a++) {
        Type init_prev_unalloc_age = 1/(1+exp(-(init_logit_prev_i[r] + (init_prev_f + init_prev_f_i[r]) * s)));
        Type init_art_unalloc_age = 1/(1+exp(-(logit_initial_art_i[r] +  logit_initial_art_mean + (init_art_f + init_art_f_i[r]) * s)));
        init_inf_no_ART.row(s * n_region * n_age + r * n_age + a) = init_pop[s * n_region * n_age + r * n_age + a] * init_substage_dist.col(0 + s) * init_prev_unalloc_age;
        if (post_scaleup) {
          init_art.row(s * n_region * n_age + r * n_age + a) = init_pop[s * n_region * n_age + r * n_age + a] * init_art_unalloc_age * init_substage_dist.col(2 + s);
          init_inf_no_ART.row(s * n_region * n_age + r * n_age + a) *= (1-init_art_unalloc_age);
        }
      }
    }
  }

  // Calculate counts of I(0) and A(0)
  matrix<Type> init_inf;
  init_inf.setZero(n_sex * n_region * n_age, n_substage[2]);
  init_inf = init_art.array() + init_inf_no_ART.array();

  REPORT(init_art);
  REPORT(init_inf);

  // Calculate log-transmission rate series
  matrix<Type> log_kappa_ts(n_ts, n_region);
  vector<Type> tmp_kappa_col(n_ts);
  vector<Type> kappa_arima(n_ts);
  vector<Type> kappa_linear(n_ts);
  vector<Type> kappa_arima_coef(kappa_rw.size());
  for (size_t r = 0; r < n_region; r++) {
    kappa_linear = kappa_r_0_i[r] + kappa_r_0 + (log_kappa_alpha_i[r] + log_kappa_alpha) * kappa_times;
    kappa_arima_coef = delta_t * (kappa_rw_coef * sqrt(1/sigma_kappa_rw) + kappa_rw_i_coef.col(r).array() * sqrt(1/sigma_kappa_rw_i));
    kappa_arima = spline_ts * kappa_arima_coef;
    log_kappa_ts.col(r) = kappa_linear + kappa_w * kappa_arima;
  }
  REPORT(kappa_t_mid_v);
  REPORT(kappa_alpha);
  REPORT(kappa_r_0_v);
  REPORT(log_kappa_ts);

  // Exponentiate kappa matrix
  matrix<Type> kappa_ts = exp(log_kappa_ts.array());

  vector<Type> art_init_sex(2);
  art_init_sex(0) = 1;
  art_init_sex(1) = exp(art_init_b_s);

  // Find ART initiation probabilities
  matrix<Type> art_init_m_rate;
  art_init_m_rate.setZero(n_ts, n_sex * n_region * n_substage[1] * n_age);

  // Predict ART initiation (alpha)
  matrix<Type> art_init_m;
  art_init_m.setZero(n_ts, n_sex * n_region * n_substage[1] * n_age);
  vector<Type> tmp_art_init(n_ts);
  vector<Type> tmp_v(n_ts);
  vector<Type> tmp_bg_mort(n_ts);
  vector<Type> hiv_exits(n_ts);
  vector<Type> art_init_ratio(n_ts);
  vector<Type> art_init_b_time(art_init_rw.size());

  vector<Type> tmp_rate(n_ts);
  matrix<Type> art_init_out(n_ts, n_region);
  matrix<Type> art_init_time_out(n_ts, n_region);
  vector<Type> tmp_sex_effect(n_ts);
  vector<Type> tmp_sex_time(spline_ts_art_init_sex.cols());

  for (size_t r = 0; r < n_region; r++) {
    art_init_b_time = art_init_rw * sqrt(1/sigma_art_init_rw) + art_init_rw_i_coef.col(r).array() * sqrt(1/sigma_art_init_rw_i);
    art_init_time_out.col(r) = art_init_design * art_init_b_time;
    tmp_art_init = art_init_time_out.col(r).array() + art_init_intercept + art_init_b_r[r] + (art_init_b_t + art_init_b_t_i[r]) * art_times;
    art_init_out.col(r) = tmp_art_init;
    tmp_v = art_design_binary.col(r);
    tmp_sex_time = art_init_b_s_t + art_init_b_s_t_i.col(r).array();
    tmp_sex_effect = spline_ts_art_init_sex * tmp_sex_time;
    for (size_t g = 0; g < n_sex; g++) {
      for (size_t a = 0; a < n_age; a++) {
        tmp_bg_mort = mort_bg.row(g * n_age + a);
        for (size_t s = 0; s < n_substage[1]; s++) {
          art_init_ratio = mort_ratio_m.row(s * n_sex + g).array();
          tmp_rate = exp((art_init_b_s + art_init_b_s_i[r] + tmp_sex_effect) * g + tmp_art_init + art_init_b_mort * log(art_init_ratio)) * tmp_v * cd4_elig.col(g * n_substage[1] + s).array();
          art_init_m_rate.col(g * n_substage[1] * n_region * n_age + s * n_region * n_age + r * n_age + a) = tmp_rate;

          hiv_exits = transition_prob.row(s * n_sex + g).array() + mort_stage.row((s+n_substage[0]) * n_age * n_sex + g * n_age + a).array();
          art_init_m.col(g * n_substage[1] * n_region * n_age + s * n_region * n_age + r * n_age + a) = (1 - (hiv_exits + tmp_bg_mort)) * (1-exp(-dt * tmp_rate));
        }
      }
    }
  }
  REPORT(art_init_m_rate);
  REPORT(tmp_art_init);
  REPORT(art_init_out);
  REPORT(art_init_time_out);

  // Integrate compartmental model
  matrix<Type> out_proj = project_EPP(n_region, n_ts, is_demog_ts, n_stage, n_nz, n_age, n_sex, dt,
    init_pop, init_inf, init_art, adjacency_m, kappa_ts, mort_stage, from_mat_i,
    to_mat_i, from_region, to_region, from_i, to_i, transm_reduction,
    to_m, to_n, entrant_m, art_init_m, eta, n_substage,
    n_transition, n_from, from_group, to_group, from_substage, stage_map,
    from_compart, transition_prob, mort_bg, substage_map, d_cd4,
    from_age, to_age, sex_IRR, pop_m, from_sex, entrant_dist, exit_m, exit_dist);

  // Report disease projection
  REPORT(out_proj);

  // Evaluate log-prior and log-posterior densities
  if (!is_simulation) {

    // Exponentiate log-variances
    Type sigma_prev_i = exp(log_sigma_prev_i);
    Type sigma_anc_delta_site = exp(log_sigma_anc_delta_site);
    Type sigma_anc_delta_b_t_i = exp(log_sigma_anc_delta_b_t_i);
    Type sigma_art_init_b_t_i = exp(log_sigma_art_init_b_t_i);
    Type sigma_initial_art_i = exp(log_sigma_initial_art_i);
    Type sigma_ART_repel = exp(log_sigma_ART_repel);
    Type sigma_ART_attract = exp(log_sigma_ART_attract);
    Type sigma_init_prev_f = exp(log_sigma_init_prev_f);
    Type sigma_init_art_f = exp(log_sigma_init_art_f);
    Type sigma_b_time_i = exp(log_sigma_b_time_i);
    Type sigma_art_init_b_r = exp(log_sigma_art_init_b_r);
    Type sigma_art_init_b_s_i = exp(log_sigma_art_init_b_s_i);

    Type sigma_kappa_t_mid = exp(log_sigma_kappa_t_mid);
    Type sigma_kappa_r_0 = exp(log_sigma_kappa_r_0);
    Type sigma_kappa_r_inf_delta = exp(log_sigma_kappa_r_inf_delta);
    Type sigma_kappa_alpha = exp(log_sigma_kappa_alpha);

    Type spline_smooth_attract = exp(log_spline_smooth_attract);
    Type ART_attract_mean = (tol_val + (1-tol_val) * exp(logit_ART_attract_mean))/(exp(logit_ART_attract_mean) + 1);
    REPORT(ART_attract_mean);

    vector<Type> prev_tested_hat;
    vector<Type> prev_pos_hat;
    vector<Type> prev_art_hat;

    prev_tested_hat.setZero(n_survey);
    prev_pos_hat.setZero(n_survey);
    prev_art_hat.setZero(n_survey);

    // Aggregate survey prevalence estimates
    for (size_t i = 0; i < n_in_survey; i++) {
      for (size_t j = 0; j < n_prev_survey_contrib[i]; j++) {
        for (size_t s = 0; s < n_stage; s++) {
          if (stage_map[s] == 0) {
            prev_tested_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
          }
          if (stage_map[s] == 1) {
            prev_tested_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
            prev_pos_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
          }
          if (stage_map[s] == 2) {
            prev_tested_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
            prev_pos_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
            prev_art_hat[prev_obs_m(i, j)] += out_proj(has_survey_i[i], s);
          }
        }
      }
    }

    // Find survey prevalence and evaluate likelihood
    vector<Type> prev_hat = prev_pos_hat / prev_tested_hat;
    for (size_t i = 0; i < prev_hat.size(); i++) {
      if (prev_tested_hat[i] == 0 | prev_pos_hat[i] == 0) {
        prev_hat[i] = tol_val;
      }
      if (prev_tested_hat[i] ==  prev_pos_hat[i]) {
        prev_hat[i] = 1-tol_val;
      }
      if (prev_in_sample[i] == 1) {
        llikelihood -= dbinom(prev_pos[i], prev_tested[i], prev_hat[i], true);
      }
    }
    REPORT(prev_hat);
    REPORT(prev_pos_hat);
    REPORT(prev_tested_hat);
    int stop_i = 1;

    // Evaluate likelhood and priors for recency data if they exist
    if (any_recent) {
      vector<Type> new_inf_hat;
      vector<Type> recent_pos_hat;
      vector<Type> recent_pop_hat;
      vector<Type> recent_suscept_hat;
      new_inf_hat.setZero(n_recent);
      recent_suscept_hat.setZero(n_recent);
      recent_pos_hat.setZero(n_recent);
      recent_pop_hat.setZero(n_recent);

      // Aggregate estimates for recency data
      for (size_t i = 0; i < n_in_recent; i++) {
        for (size_t j = 0; j < n_recent_survey_contrib[i]; j++) {
          new_inf_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], n_stage) * out_proj(has_recent_i[i], 0);
          for (size_t s = 0; s < n_stage; s++) {
            if (stage_map[s] == 0) {
              recent_suscept_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
              recent_pop_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
            } else if (stage_map[s] == 1) {
              recent_pop_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
              recent_pos_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
            } else if (stage_map[s] == 2) {
              recent_pop_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
              recent_pos_hat[recent_obs_m(i, j)] += out_proj(has_recent_i[i], s);
            }
          }
        }
      }
      // Calculate appropriate incidence rate and probability of recency given incidence and prevalence
      vector<Type> incidence_hat = -log(1 - new_inf_hat / recent_suscept_hat) / dt;
      vector<Type> prob_recent_hat = incidence_hat / (recent_pos_hat/recent_pop_hat) * (1 - (recent_pos_hat/recent_pop_hat)) * MDRI;

      for (size_t i = 0; i < recent_pop_hat.size(); i++) {
        if (recent_pop_hat[i] == 0 | incidence_hat[i] == 0 | prob_recent_hat[i] == 0 | new_inf_hat[i] == 0) {
          prob_recent_hat[i] = tol_val;
        }
        if (recent_in_sample[i] == 1) {
          llikelihood -= dbinom(recent_pos[i], recent_tested[i], prob_recent_hat[i], true);
        }

      }


      REPORT(prob_recent_hat);
      REPORT(incidence_hat);
      REPORT(recent_pop_hat);
      REPORT(recent_pos_hat);
      REPORT(new_inf_hat);
      REPORT(recent_suscept_hat);
    }

    // Evaluate likelhood and priors for ART testing data if they exist
    if (any_art_test) {
      vector<Type> art_test_tested_hat;
      vector<Type> art_test_pos_hat;

      art_test_tested_hat.setZero(n_art_test);
      art_test_pos_hat.setZero(n_art_test);

      for (size_t i = 0; i < n_in_art_test; i++) {
        for (size_t j = 0; j < n_art_test_survey_contrib[i]; j++) {
          for (size_t s = 0; s < n_stage; s++) {
            if (stage_map[s] == 0) {

            } else if (stage_map[s] == 1) {
              art_test_tested_hat[art_test_obs_m(i, j)] += out_proj(has_art_test_i[i], s);
            } else if (stage_map[s] == 2) {
              art_test_tested_hat[art_test_obs_m(i, j)] += out_proj(has_art_test_i[i], s);
              art_test_pos_hat[art_test_obs_m(i, j)] += out_proj(has_art_test_i[i], s);
            }
          }
        }
      }
      vector<Type> art_test_hat = art_test_pos_hat / art_test_tested_hat;
      for (size_t i = 0; i < art_test_tested_hat.size(); i++) {
        if (art_test_tested_hat[i] == 0) {
          art_test_hat[i] = tol_val;
        }
        if (art_test_tested_hat[i] == art_test_pos_hat[i]) {
          art_test_hat[i] = 1-tol_val;
        }
        if (art_test_in_sample[i] == 1) {
          llikelihood -= dbinom(art_pos[i], art_tested[i], art_test_hat[i], true);
        }
      }
      REPORT(art_test_hat);
    }

    // Evaluate likelhood for ANC data if they exist
    if (any_anc) {
      vector<Type> anc_tested_hat;
      vector<Type> anc_pos_hat;

      anc_tested_hat.setZero(n_anc);
      anc_pos_hat.setZero(n_anc);

      for (size_t i = 0; i < n_in_anc; i++) {
        for (size_t j = 0; j < n_anc_survey_contrib[i]; j++) {
          for (size_t s = 0; s < n_stage; s++) {
            if (stage_map[s] == 0) {
              anc_tested_hat[anc_obs_m(i, j)] += out_proj(has_anc_i[i], s);

            } else if (stage_map[s] == 1) {
              anc_tested_hat[anc_obs_m(i, j)] += out_proj(has_anc_i[i], s);
              anc_pos_hat[anc_obs_m(i, j)] += out_proj(has_anc_i[i], s);

            } else if (stage_map[s] == 2) {
              anc_tested_hat[anc_obs_m(i, j)] += out_proj(has_anc_i[i], s);
              anc_pos_hat[anc_obs_m(i, j)] += out_proj(has_anc_i[i], s);
            }
          }
        }
      }
      vector<Type> unadj_anc_hat = anc_pos_hat / anc_tested_hat;

      // Adding site-specific random effects
      vector<Type> logit_anc_hat = log(unadj_anc_hat/(1-unadj_anc_hat));
      for (size_t i = 0; i < n_anc; i++) {
        logit_anc_hat[i] += logit_anc_delta_site[site_i[i]] + (logit_anc_delta_b_t + logit_anc_delta_b_t_i[site_i[i]]) * anc_ts[i];
      }
      vector<Type> anc_hat = 1/(1+exp(-logit_anc_hat)) + tol_val;

      vector<Type> anc_corr = 1/(1+exp(-logit_anc_corr));
      vector<Type> a(n_anc);
      for (size_t i = 0; i < n_anc; i++) {
        a[i] = (1-anc_corr[anc_agg_i[i]])/anc_corr[anc_agg_i[i]] * anc_hat[i];
      }
      vector<Type> b = a * (1-anc_hat)/anc_hat;

      for (size_t i = 0; i < anc_hat.size(); i++) {
        if(anc_hat[i]  > (1-tol_val)) {
          anc_hat[i] = 1 - tol_val;
        }
        if (anc_in_sample[i] == 1) {
          if (anc_likelihood == 1) {
            llikelihood -= dbinom(anc_pos[i], anc_tested[i], anc_hat[i], true);
          } else if (anc_likelihood == 2) {
            llikelihood -= dbbinom(anc_pos[i], anc_tested[i], a[i], b[i]);
            // nll -= dbbinom(y[i], n_trial[i], a[i], b[i]) * in_sample[i];
          }
        }
      }
      REPORT(a);
      REPORT(b);
      lprior -= dnorm(logit_anc_corr, Type(-1), Type(1), true).sum();

      REPORT(logit_anc_hat);
      REPORT(anc_pos_hat);
      REPORT(anc_tested_hat);
      REPORT(anc_hat);
      REPORT(anc_corr);
    }

    int count_i = 1;
    // Priors for inital prevalence
    lprior -= sum(dnorm(init_logit_prev_i, init_logit_prev_mean, sigma_prev_i, true));
    Type nat_prev_hat = (init_pop_share/(1+exp(-init_logit_prev_i))).sum();
    lprior -= dnorm(nat_prev_hat, init_prev, Type(0.0025), true);
    ADREPORT(nat_prev_hat);
    lprior -= (log(init_pop_share) - init_logit_prev_i - 2 * log(1 + exp(-init_logit_prev_i))).sum();
    lprior -= dnorm(init_logit_prev_mean, Type(-2), Type(1), true);
    lprior -= log(Type(2.0)) + dnorm(sigma_prev_i, Type(0.0), Type(1), true);
    // Change-of-variables adjustment for sigma_prev
    lprior -= log_sigma_prev_i;

    // Priors on p-spline coefficients
    // Centered parameterization isn't currently working for some reason.
    vector<Type> tmp_v;
    vector<Type> prev_v;
    vector<Type> diff_v;
    Type diff_val;
    Type prev_val;

    lprior -= dnorm(kappa_r_0, Type(-2.3), Type(1), true);
    lprior -= dnorm(kappa_r_0_i, Type(0), sigma_kappa_r_0, true).sum();
    lprior -= dnorm(sigma_kappa_r_0, Type(0), Type(1), true);
    lprior -= log_sigma_kappa_r_0;

    lprior -= dnorm(log_kappa_alpha, Type(0), Type(5), true);
    lprior -= sum(dnorm(log_kappa_alpha_i, Type(0.0), sigma_kappa_alpha, true));
    lprior -= dnorm(sigma_kappa_alpha, Type(0.0), Type(2.0), true);
    lprior -= log_sigma_kappa_alpha;

    Type rw_sd = sqrt(1-phi_kappa_rw * phi_kappa_rw);
    if (use_autoregressive == 0) {
      rw_sd = 1.0;
    }
    vector<Type> kappa_rw_diff(kappa_rw.size()-diff_ord);

    kappa_rw_diff.setZero(kappa_rw.size()-diff_ord);

    lprior -= dnorm(eps_kappa_rw, Type(0), Type(1), true).sum();

    for (size_t i = diff_ord; i < kappa_rw.size(); i++) {
      if (diff_ord == 0) {
        kappa_rw_diff[i-diff_ord] = kappa_rw[i];
      }
      if (diff_ord == 1) {
        kappa_rw_diff[i-diff_ord] = kappa_rw[i] - kappa_rw[i-1];
      }
      if (diff_ord == 2) {
        kappa_rw_diff[i-diff_ord] = (kappa_rw[i] - kappa_rw[i-1]) - (kappa_rw[i-1] - kappa_rw[i-2]);
      }
      if (i > diff_ord & model_innovations == 0) {
        lprior -= dnorm(kappa_rw.sum(), Type(0), Type(0.01) * kappa_rw.size(), true);
        lprior -= dnorm(kappa_rw_diff[i-diff_ord], phi_kappa_rw * kappa_rw_diff[i-diff_ord-1], rw_sd, true);
      }
    }

    nll -= dbeta(phi_kappa_rw_i, Type(1), Type(0.2), true);
    nll -= logit_phi_kappa_rw_i - 2 * log(1 + exp(logit_phi_kappa_rw_i));

    lprior -= dlgamma(log_sigma_kappa_rw_i, Type(1), Type(20000), true);

    vector<Type> tmp_kappa_rw;
    vector<Type> tmp_eps_kappa_rw;
    Type rw_sd_i = sqrt(1-phi_kappa_rw_i *phi_kappa_rw_i);
    if (use_autoregressive == 0) {
      rw_sd_i = 1.0;
    }

    for (size_t i = 0; i < n_region; i++) {
      tmp_kappa_rw = kappa_rw_i.col(i);
      tmp_eps_kappa_rw = eps_kappa_rw_i.col(i);
      kappa_rw_diff.setZero(tmp_kappa_rw.size()-diff_ord);

      lprior -= dnorm(tmp_eps_kappa_rw, Type(0), Type(1), true).sum();

      for (size_t i = diff_ord; i < tmp_kappa_rw.size(); i++) {
        if (diff_ord == 0) {
          kappa_rw_diff[i-diff_ord] = tmp_kappa_rw[i];
        }
        if (diff_ord == 1) {
          kappa_rw_diff[i-diff_ord] = tmp_kappa_rw[i] - tmp_kappa_rw[i-1];
        }
        if (diff_ord == 2) {
          kappa_rw_diff[i-diff_ord] = (tmp_kappa_rw[i] - tmp_kappa_rw[i-1]) - (tmp_kappa_rw[i-1] - tmp_kappa_rw[i-2]);
        }
        if (i > diff_ord & model_innovations == 0) {
          lprior -= dnorm(tmp_kappa_rw.sum(), Type(0), Type(0.01) * tmp_kappa_rw.size(), true);
          lprior -= dnorm(kappa_rw_diff[i-diff_ord], phi_kappa_rw_i * kappa_rw_diff[i-diff_ord-1], rw_sd_i, true);
        }
      }
    }

    nll -= dbeta(phi_kappa_rw_i, Type(1), Type(0.2), true);
    nll -= logit_phi_kappa_rw_i - 2 * log(1 + exp(logit_phi_kappa_rw_i));

    lprior -= dlgamma(log_sigma_kappa_rw_i, Type(1), Type(20000), true);

    // Priors on ANC site effects
    lprior -= dnorm(logit_anc_delta, Type(0), Type(5), true);
    lprior -= sum(dnorm(logit_anc_delta_site, logit_anc_delta, sigma_anc_delta_site, true));
    lprior -= log(Type(2.0)) + dnorm(sigma_anc_delta_site, Type(0.0), Type(1.0), true);
    lprior -= log_sigma_anc_delta_site;
    lprior -= dnorm(logit_anc_delta_b_t, Type(0), Type(5), true);
    lprior -= sum(dnorm(logit_anc_delta_b_t_i, Type(0), sigma_anc_delta_b_t_i, true));
    lprior -= dnorm(sigma_anc_delta_b_t_i, Type(0.0), Type(1.0), true);
    lprior -= log_sigma_anc_delta_b_t_i;

    // Priors on ART initiation logistic model coefficients
    lprior -= dnorm(art_init_intercept, Type(0), Type(5.0), true);
    lprior -= dnorm(art_init_rw.sum(), Type(0), Type(0.0001), true);

    vector<Type> art_init_rw_diff(art_init_rw.size()-1);
    for (size_t i = 1; i < art_init_rw.size(); i++) {
      art_init_rw_diff[i-1] = art_init_rw[i] - art_init_rw[i-1];
    }
    lprior += AR1(Type(phi_art_init_rw))(art_init_rw_diff);

    nll -= dnorm(logit_phi_art_init_rw, Type(0), Type(sqrt(1/0.15)), true);
    lprior -= dlgamma(log_sigma_art_init_rw, Type(1), Type(20000), true);
    vector<Type> tmp_art_init_rw_i;
    vector<Type> tmp_eps_art_init_rw;
    for (size_t i = 0; i < n_region; i++) {
      tmp_eps_art_init_rw = eps_art_init_rw_i.col(i);
      lprior -= dnorm(tmp_eps_art_init_rw, Type(0), Type(1), true).sum();

      for (size_t i = 1; i < tmp_art_init_rw_i.size(); i++) {
        art_init_rw_diff[i-1] = tmp_art_init_rw_i[i] - tmp_art_init_rw_i[i-1];
      }
      if (model_innovations_art == 0) {
        lprior -= dnorm(tmp_art_init_rw_i.sum(), Type(0), Type(0.0001), true);
        lprior += AR1(Type(phi_art_init_rw_i))(art_init_rw_diff);
      }
    }

    nll -= dnorm(logit_phi_art_init_rw_i, Type(0), Type(sqrt(1/0.15)), true);
    lprior -= dlgamma(log_sigma_art_init_rw_i, Type(1), Type(20000), true);

    lprior -= sum(dnorm(art_init_b_r, Type(0.0), sigma_art_init_b_r, true));
    lprior -= dnorm(sigma_art_init_b_r, Type(0.0), Type(2.0), true);
    lprior -= log_sigma_art_init_b_r;

    lprior -= dnorm(art_init_b_t, Type(0), Type(5), true);
    lprior -= sum(dnorm(art_init_b_t_i, Type(0.0), sigma_art_init_b_t_i, true));
    lprior -= dnorm(sigma_art_init_b_t_i, Type(0.0), Type(2.0), true);
    lprior -= log_sigma_art_init_b_t_i;

    // lprior -= dnorm(art_init_b_s_t, Type(0), Type(5), true).sum();
    lprior -= dnorm(art_init_b_s_t[0], Type(0), Type(0.01), true);
    for (size_t i = 1; i < art_init_b_s_t.size(); i++) {
      lprior -= dnorm(art_init_b_s_t[i]-art_init_b_s_t[i-1], Type(0), exp(log_sigma_art_init_b_s_t), true);
    }
    lprior -= dnorm(exp(log_sigma_art_init_b_s_t), Type(0.0), Type(2.0), true);
    lprior -= log_sigma_art_init_b_s_t;

    lprior -= dnorm(art_init_b_s, Type(0), Type(5), true);
    lprior -= sum(dnorm(art_init_b_s_i, Type(0.0), sigma_art_init_b_s_i, true));
    lprior -= dnorm(sigma_art_init_b_s_i, Type(0.0), Type(2.0), true);
    lprior -= log_sigma_art_init_b_s_i;

    vector<Type> tmp_b_row;
    tmp_b_row = art_init_b_s_t_i.row(0);
    lprior -= sum(dnorm(tmp_b_row, Type(0), Type(0.01), true));
    for (size_t i = 1; i < art_init_b_s_t_i.rows(); i++) {
      tmp_b_row = art_init_b_s_t_i.row(i)-art_init_b_s_t_i.row(i-1);
      lprior -= sum(dnorm(tmp_b_row, Type(0), exp(log_sigma_art_init_b_s_t_i), true));
    }
    lprior -= dnorm(exp(log_sigma_art_init_b_s_t_i), Type(0.0), Type(2.0), true);
    lprior -= log_sigma_art_init_b_s_t_i;

    lprior -= dnorm(art_init_b_mort, Type(0), Type(1), true);
    lprior -= dnorm(art_init_b_elig_s, Type(0), Type(1), true);
    lprior -= dnorm(log_art_init_b_elig, Type(0.0), Type(5.0), true);

    // Priors on inital ART coverage
    lprior -= sum(dnorm(logit_initial_art_i, Type(0.0), sigma_initial_art_i, true));
    lprior -= dnorm(logit_initial_art_mean, Type(0.0), Type(5.0), true);
    lprior -= log(Type(2.0)) + dnorm(sigma_initial_art_i, Type(0.0), Type(1.0), true);
    // COV adjustment
    lprior -= log_sigma_initial_art_i;

    // Predict ART attendance
    vector<Type> tmp_OR;
    vector<int> tmp_nb_i;
    vector<Type> tmp_dist_v;
    Type cumul_mass;
    int sneaky_counter = 0;
    vector<Type> tmp_log_ART_repel;
    tmp_log_ART_repel.setZero(n_region);
    vector<Type> tmp_log_ART_attract;
    tmp_log_ART_attract.setZero(n_region);
    matrix<Type> tmp_b_time_i;
    tmp_b_time_i.setZero(b_time_i.rows(), n_region);
    vector<Type> repel_prior;
    repel_prior.setZero(n_region);
    int counter = 0;
    Type tmp_attract_mean;
    for (size_t i = 0; i < n_region; i++) {
      if (n_nb[i] > 1) {
        tmp_log_ART_repel[i] = log_ART_repel[counter];
        tmp_attract_mean = 1/(1+exp(-(logit_ART_attract_mean + beta_dist * log_ART_repel[counter])));
        tmp_log_ART_attract[i] = log_ART_attract[counter];
        repel_prior[i] = log(tmp_attract_mean/((n_nb[i]-1) - (n_nb[i]-1) * tmp_attract_mean));
        tmp_b_time_i.col(i) = b_time_i.col(counter);
        counter += 1;
      }
    }
    REPORT(repel_prior);
    REPORT(tmp_log_ART_repel);
    REPORT(tmp_log_ART_attract);
    typedef Eigen::Triplet<Type> T;
    typedef Eigen::SparseMatrix<Type> S;
    Eigen::SparseMatrix<Type> art_shares(n_region, n_region);

    std::vector<T> tripletList;
    tripletList.reserve(n_nb.sum());
    vector<Type> tmp_art_share;
    std::vector<S> share_v;
    matrix<Type> prob_stay(n_region, n_ts);
    vector<Type> time_contrib = spline_m_attract * b_time;
    matrix<Type> time_contrib_i(n_region, spline_m_attract.rows());
    vector<Type> test_v(n_ts);
    for (size_t r = 0; r < n_region; r++) {
      time_contrib_i.row(r) = (spline_m_attract * tmp_b_time_i.col(r)).transpose();
    }
    Type tmp_attract_term = 0;
    matrix<Type> logit_attract_means;
    logit_attract_means.setZero(n_region,n_ts);
    matrix<Type> attract_means;
    attract_means.setZero(n_region,n_ts);
    for (size_t t = 0; t < n_ts; t++) {
      tripletList.clear();
      for (size_t r = 0; r < n_region; r++) {
        tmp_OR.setZero(n_region);
        cumul_mass = 0;
        tmp_nb_i = wide_nb_m.row(r);
        tmp_dist_v = dist_m.row(r);
        for (size_t i = 0; i < n_nb[r]; i++) {
          if (tmp_nb_i[i] == r) {
            cumul_mass += tmp_OR[tmp_nb_i[i]] = 1.0;
          } else {
            cumul_mass += tmp_OR[tmp_nb_i[i]] = exp(log_ART_repel_mean + tmp_log_ART_repel[tmp_nb_i[i]] + tmp_log_ART_attract[i] + time_contrib[t] + time_contrib_i(tmp_nb_i[i], t));
            sneaky_counter += 1;
          }
        }
        tmp_art_share = tmp_OR / cumul_mass;
        prob_stay(r, t) = tmp_art_share[r];
        for (size_t i = 0; i < n_nb[r]; i++) {
          tripletList.push_back(T(tmp_nb_i[i], r, tmp_art_share[tmp_nb_i[i]]));
        }
      }
      art_shares.setFromTriplets(tripletList.begin(), tripletList.end());
      share_v.push_back(art_shares);
    }
    REPORT(prob_stay);
    REPORT(art_shares);
    REPORT(attract_means);
    REPORT(logit_attract_means);

    matrix<Type> art_ts(n_region, n_ts);
    vector<Type> tmp_art_count(n_region);
    for (size_t t = 0; t < n_ts; t++) {
      tmp_art_count.setZero(n_region);
      for (size_t r = 0; r < n_region; r++) {
        for (size_t g = 0; g < n_sex; g++) {
          for (size_t s = 0; s < n_stage; s++) {
            if (stage_map[s] == 2) {
              tmp_art_count[r] += out_proj.block(t * n_region * n_age * n_sex + g * n_age * n_region + r * n_age, s, n_age, 1).sum();
            }
          }
        }
      }
      art_ts.col(t) = share_v[t] * tmp_art_count;
    }
    REPORT(art_ts);
    vector<Type> art_count_hat;

    art_count_hat.setZero(n_art_count);

    // Aggregating ART count estimates
    for (size_t i = 0; i < n_in_art_count; i++) {
      for (size_t j = 0; j < n_art_count_survey_contrib[i]; j++) {
        art_count_hat[art_count_obs_m(i, j)] += art_ts(art_count_region_i[i], art_count_ts_i[i]);
      }
    }
    for (size_t i = 0; i < art_count_hat.size(); i++) {
      if (art_count_hat[i] <= 0) {
        art_count_hat[i] = Type(1e-4);
      }
    }
    // Evaluating likelihood for ART count seriess
    Type omega = exp(log_omega) + 1e-12;
    Type theta = exp(log_overdispersion) + 1e-12;
    vector<Type> mu_sq = art_count_hat.array() * art_count_hat.array();

    // ART programme data likelihood
    vector<Type> nbinom_vars = art_count_hat * (1 + omega + art_count_hat * theta);
    vector<Type> probs = art_count_hat / nbinom_vars;
    vector<Type> sizes = mu_sq / (nbinom_vars - art_count_hat);
    REPORT(nbinom_vars);
    REPORT(omega);
    REPORT(theta);
    REPORT(art_count_hat);
    REPORT(probs);
    REPORT(sizes);

    lprior -= log(Type(2.0)) + dnorm(exp_eta, Type(0.00), Type(5), 1);
    lprior -= log_eta;
    vector<Type> art_count_nll;
    if (art_ll == 1) {
      for (size_t i = 0; i < art_count.size(); i++) {
        if (art_count_in_sample[i] == 1) {
          llikelihood -= dnbinom(art_count[i], sizes[i], probs[i], true);
        }
      }
      lprior -= dnorm(log_omega, Type(0.0), Type(2), true);

      lprior -= dnorm(log_overdispersion, Type(0), Type(2), true);

    } else if (art_ll == 2) {
      for (size_t i = 0; i < art_count.size(); i++) {
        if (art_count_in_sample[i] == 1) {
          llikelihood -= dpois(art_count[i], art_count_hat[i], true);
        }
      }
    }
    // Priors for ART attendance model
    lprior -= sum(dnorm(log_ART_attract, Type(0.0), sigma_ART_attract, true));
    lprior -= dnorm(logit_ART_attract_mean, exp_mean, Type(1.0), true);
    lprior -= dnorm(sigma_ART_attract, Type(0.0), Type(1.0), true);
    lprior -= log_sigma_ART_attract;
    lprior -= dnorm(beta_dist, Type(0.0), Type(1), true);

    lprior -= sum(dnorm(log_ART_repel, Type(0.0), sigma_ART_repel, true));
    lprior -= dnorm(log_ART_repel_mean, Type(-3), Type(0.1), true);
    lprior -= dnorm(sigma_ART_repel, Type(0.0), Type(2), true);
    lprior -= log_sigma_ART_repel;

    lprior -= dnorm(init_prev_f, Type(0.0), Type(1), true);
    lprior -= sum(dnorm(init_prev_f_i, Type(0.0), sigma_init_prev_f, true));
    lprior -= dnorm(sigma_init_prev_f, Type(0), Type(2), true);
    lprior -= log_sigma_init_prev_f;

    lprior -= dnorm(init_art_f, Type(0.0), Type(5), true);
    lprior -= sum(dnorm(init_art_f_i, Type(0.0), sigma_init_art_f, true));
    lprior -= dnorm(sigma_init_art_f, Type(0), Type(2), true);
    lprior -= log_sigma_init_art_f;

    lprior -= dnorm(b_time, Type(0), Type(5), true).sum();

    vector<Type> tmp_b_v(b_time_i.rows());
    for (size_t r = 0; r < b_time_i.cols(); r++) {
      lprior -= dnorm(b_time_i(0, r), Type(0.0), sigma_b_time_i, true);
      for (size_t i = 1; i < b_time_i.rows(); i++) {
        lprior -= dnorm(b_time_i(i, r), b_time_i(i-1, r), sigma_b_time_i, true);
      }
    }

    lprior -= dnorm(sigma_b_time_i, Type(0.0), Type(1.0), true);
    lprior -= log_sigma_b_time_i;

    lprior -= dnorm(spline_smooth_attract, Type(0.0), Type(1.0), true);
    lprior -= log_spline_smooth_attract;
    REPORT(llikelihood);
    REPORT(lprior);

    nll += llikelihood + lprior;

    ADREPORT(phi_kappa_rw);
    ADREPORT(phi_kappa_rw_i);
  }
  // Return NLL to TMB
  REPORT(nll);
  return nll;
}
