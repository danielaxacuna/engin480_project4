# Delivery 3: Hybrid Power Plants by hysdesign 


import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath



examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
examples_sites



name = 'France_good_wind'
ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]


nput_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]


input_ts_fn = examples_filepath + ex_site['input_ts_fn'].values[0]
input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)

required_cols = [col for col in input_ts.columns if 'WD' not in col]
input_ts = input_ts.loc[:, required_cols]

print(input_ts)


sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]

with open(sim_pars_fn) as file:
    sim_pars = yaml.load(file, Loader=yaml.FullLoader)

print(sim_pars_fn)
sim_pars


############
############


rotor_diameter_m = 220
hub_height_m = 150
wt_rated_power_MW = 13
surface_tilt_deg = 20
surface_azimuth_deg = 135
DC_AC_ratio = 1.2



############
############




hpp = hpp_model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        rotor_diameter_m = rotor_diameter_m,
        hub_height_m = hub_height_m,
        wt_rated_power_MW = wt_rated_power_MW,
        surface_tilt_deg = surface_tilt_deg,
        surface_azimuth_deg = surface_azimuth_deg,
        DC_AC_ratio = DC_AC_ratio,
        num_batteries = 12,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn,
)



############
############



start = time.time()

Nwt = 60
wind_MW_per_km2 = 7
solar_MW = 130
b_P = 20
b_E_h  = 3
cost_of_batt_degr = 7
clearance = hub_height_m - rotor_diameter_m / 2
sp = 4 * wt_rated_power_MW * 10 ** 6 / np.pi / rotor_diameter_m ** 2

x = [# Wind plant design
    clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    # PV plant design
    solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    # Energy storage & EMS price constrains
    b_P, b_E_h, cost_of_batt_degr]

outs = hpp.evaluate(*x)

hpp.print_design(x, outs)

end = time.time()

print(f'exec. time [min]:', (end - start)/60 )




###########
###########


b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
b_t = hpp.prob.get_val('ems.b_t')
price_t = hpp.prob.get_val('ems.price_t')

wind_t = hpp.prob.get_val('ems.wind_t')
solar_t = hpp.prob.get_val('ems.solar_t')
hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')
grid_MW = hpp.prob.get_val('ems.G_MW')

n_days_plot = 14

# Plot battery and price signals
plt.figure(figsize=[12, 4])
plt.plot(price_t[:24 * n_days_plot], label='Price [$]')
plt.plot(b_E_SOC_t[:24 * n_days_plot], label='State of Charge [MWh]')
plt.plot(b_t[:24 * n_days_plot], label='Battery Power [MW]')
plt.xlabel('Time [hours]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, fancybox=True, shadow=True)

# Plot power generation and curtailment
plt.figure(figsize=[12, 4])
plt.plot(wind_t[:24 * n_days_plot], label='Wind Power [MW]')
plt.plot(solar_t[:24 * n_days_plot], label='PV Power [MW]')
plt.plot(hpp_t[:24 * n_days_plot], label='HPP Power Output [MW]')
plt.plot(hpp_curt_t[:24 * n_days_plot], label='HPP Curtailed [MW]')
plt.axhline(grid_MW, label='Grid Capacity [MW]', color='k', linestyle='--')
plt.xlabel('Time [hours]')
plt.ylabel('Power [MW]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=True, shadow=True)

# Show plots
plt.show()




############
###########



# Step 1: Simulation time and age
N_life = hpp.sim_pars['N_life']          # Number of years
life_h = int(N_life * 365 * 24)          # Total hours of simulation
age = np.arange(life_h) / (24 * 365)     # Age in years, hourly resolution

# Step 2: Extract results
SoH = hpp.prob.get_val('battery_degradation.SoH')
SoH_all = np.copy(hpp.prob.get_val('battery_loss_in_capacity_due_to_temp.SoH_all'))

wind_t_ext = hpp.prob.get_val('ems_long_term_operation.wind_t_ext')
wind_t_ext_deg = hpp.prob.get_val('ems_long_term_operation.wind_t_ext_deg')
solar_t_ext = hpp.prob.get_val('ems_long_term_operation.solar_t_ext')
solar_t_ext_deg = hpp.prob.get_val('ems_long_term_operation.solar_t_ext_deg')
hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')

# Step 3: Ensure all arrays are of same (valid) length
min_len = min(
    len(age), len(SoH), len(SoH_all),
    len(wind_t_ext), len(wind_t_ext_deg),
    len(solar_t_ext), len(solar_t_ext_deg),
    len(hpp_t), len(hpp_t_with_deg)
)

age = age[:min_len]
SoH = SoH[:min_len]
SoH_all = SoH_all[:min_len]
wind_t_ext = wind_t_ext[:min_len]
wind_t_ext_deg = wind_t_ext_deg[:min_len]
solar_t_ext = solar_t_ext[:min_len]
solar_t_ext_deg = solar_t_ext_deg[:min_len]
hpp_t = hpp_t[:min_len]
hpp_t_with_deg = hpp_t_with_deg[:min_len]

# Step 4: Construct DataFrame
index = pd.date_range(start='2023-01-01', periods=min_len, freq='1H')
df = pd.DataFrame({
    'wind_t_ext': wind_t_ext,
    'wind_t_ext_deg': wind_t_ext_deg,
    'solar_t_ext': solar_t_ext,
    'solar_t_ext_deg': solar_t_ext_deg,
    'hpp_t': hpp_t,
    'hpp_t_with_deg': hpp_t_with_deg
}, index=index)

# Step 5: Drop NaNs if any
df = df.dropna()

# Step 6: Compute yearly averages
df_year = df.groupby(df.index.year).mean()
df_year['age'] = np.arange(len(df_year)) + 0.5

# Step 7: Efficiency ratios
df_year['eff_wind_ts_deg'] = df_year['wind_t_ext_deg'] / df_year['wind_t_ext']
df_year['eff_solar_ts_deg'] = df_year['solar_t_ext_deg'] / df_year['solar_t_ext']
df_year['eff_hpp_ts_deg'] = df_year['hpp_t_with_deg'] / df_year['hpp_t']

# Step 8: Plotting
plt.figure(figsize=[12, 5])
plt.plot(df_year['age'], df_year['eff_wind_ts_deg'], label='Wind degradation')
plt.plot(df_year['age'], df_year['eff_solar_ts_deg'], label='Solar degradation')
plt.plot(df_year['age'], df_year['eff_hpp_ts_deg'], '--', label='HPP degradation')
plt.plot(age, SoH, label='Battery degradation (SoH)')
plt.plot(age, SoH_all, label='Battery + Temp loss', alpha=0.5)

plt.xlabel('Age [years]')
plt.ylabel('Degradation metrics')
plt.title('Long-Term Degradation of System Components')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#####################
####################


#solar_t = hpp.prob.get_val('ems.solar_t_ext')
b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')

b_E_SOC_t_with_deg = hpp.prob.get_val('ems_long_term_operation.b_E_SOC_t_with_deg')
hpp_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')
hpp_curt_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_curt_t_with_deg')

price_t_ext = hpp.prob.get_val('ems_long_term_operation.price_t_ext')

# Plot the HPP operation in the 7th year (with and without degradation)
n_start = int(24*365*7.2)
n_days_plot = 14

plt.figure(figsize=[12,4])

plt.plot(price_t_ext[n_start:n_start+24*n_days_plot], label='price')

plt.plot(b_E_SOC_t[n_start:n_start+24*n_days_plot], label='SoC [MWh]')
plt.plot(b_E_SOC_t_with_deg[n_start:n_start+24*n_days_plot], label='SoC with degradation [MWh]')
plt.xlabel('time [hours]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=0, shadow=0)

plt.figure(figsize=[12,4])
plt.plot(hpp_t[n_start:n_start+24*n_days_plot], label='HPP')
plt.plot(hpp_t_with_deg[n_start:n_start+24*n_days_plot], label='HPP with degradation')

plt.plot(hpp_curt_t[n_start:n_start+24*n_days_plot], label='HPP curtailed')
plt.plot(hpp_curt_t_with_deg[n_start:n_start+24*n_days_plot], label='HPP curtailed with degradation')

plt.axhline(grid_MW, label='Grid MW', color='k')
plt.xlabel('time [hours]')
plt.ylabel('Power [MW]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=6, fancybox=0, shadow=0)



###########
##########


# Scenario A (baseline)
cost_of_battery_P_fluct_in_peak_price_ratio = 0.0
x = [clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
     solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
     b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio]
outs = hpp.evaluate(*x)
SoH = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

# Scenario B
cost_of_battery_P_fluct_in_peak_price_ratio_B = 5
x = [clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
     solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
     b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio_B]
outs = hpp.evaluate(*x)
SoH_B = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

# Scenario C (this line was incorrect before)
cost_of_battery_P_fluct_in_peak_price_ratio_C = 20
x = [clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
     solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
     b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio_C]
outs = hpp.evaluate(*x)
SoH_C = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

# Create time axis
life_h = len(SoH)
age = np.arange(life_h) / (24 * 365)  # convert hours to years

# Plotting
plt.figure(figsize=[12, 3])
plt.plot(age, SoH, label=r'$C_b[f]_0=0$')
plt.plot(age, SoH_B, label=rf'$C_b[f]_B$={cost_of_battery_P_fluct_in_peak_price_ratio_B}')
plt.plot(age, SoH_C, label=rf'$C_b[f]_C$={cost_of_battery_P_fluct_in_peak_price_ratio_C}')
plt.plot(age, 0.7 * np.ones_like(age), label=r'$\min(1-L) = 0.7$', color='r', alpha=0.5)

plt.xlabel(r'Age [years]')
plt.ylabel(r'Battery State of Health, $1-L(t)$ [-]')
plt.legend(title='Cost of Battery Fluctuations',
           loc='upper center', bbox_to_anchor=(0.5, 1.27),
           ncol=3, fancybox=True, shadow=False)
plt.grid(True)
plt.tight_layout()
plt.show()




end = time.time()

print(f'exec. time [min]:', (end - start)/60 )
