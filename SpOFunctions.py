import numpy as np
import pandas as pd
import csv

sleep_duration = 8 # hours
sampling_rate = 100 # 100 hz
n_samples = int(sleep_duration * 3600 / sampling_rate)
time_vector = np.arange(n_samples) * sampling_rate / 3600 # vector of the time 0 -> 8

def generate_healthy(n_samples: int, time_h: np.ndarray) ->pd.DataFrame:
    """
    Generates a healthy SpO₂ pattern: stable, 95-100%, with minor, short dips.
    Returns DataFrame of results with time and spo over time.
    """
    print("Generating Healthy SpO₂ Pattern Data...")

    # Base level: Steady 96-98%
    base_line = np.full(n_samples, 97.0)

    # generate background noise (random fluctuations)
    noise = np.random.normal(0, 0.5, n_samples)

    # generate random, but natural, dips
    dip_amplitudes = np.random.uniform(low=-4.0, high=-2.0, size=20)
    dip_locations = np.random.randint(0, n_samples, size=20)
    dip_width = int(60 / sampling_rate)  # Dips last about 1 minute

    transient_dips = np.zeros(n_samples)

    # iterate through dips and create data
    for amp, loc in zip(dip_amplitudes, dip_locations):
        start = max(0, loc - dip_width)
        end = min(n_samples, loc + dip_width)

        # Use a gaussian window function to create a smooth, short dip
        window_size = end - start
        if window_size > 0:
            # creates a smooth exponential line that smooths itself evenly across -2 and 2
            window = np.exp(-0.5 * (np.linspace(-2, 2, window_size) ** 2))
            transient_dips[start:end] += amp * window # combine the dip

    # spo is all the data combined
    spo2_raw = base_line + noise + transient_dips

    # Then, clip it to be within a realistic range
    spo2_final = np.clip(spo2_raw, 93.0, 100.0)

    # save time data as a dataframe for later use!
    results = pd.DataFrame({'timestamp': time_h, 'spo2_percent': spo2_final})
    results.to_csv('healthy_spo_dummy_results.csv', index=False)
    return results


def generate_abnormal_spo2(n_samples: int, time_h: np.ndarray) -> pd.DataFrame:
    """
    Generates an Abnormal SpO₂ pattern:
    repeated, cyclical desaturations below 90%.
    Returns DataFrame with time and spo.
    """
    print("Generating Abnormal SpO₂ Pattern Data...")

    # base level: Stable 95.5%
    base_line = np.full(n_samples, 95.5)

    # generate background noise in the data
    noise = np.random.normal(0, 0.4, n_samples)

    # simulate repetitive desaturation events (DI of ~30/hr)
    number_of_events = 240
    event_interval = n_samples / number_of_events
    abnormal_dips = np.zeros(n_samples)

    # create data for each event
    for i in range(number_of_events):
        center_index = int(i * event_interval + np.random.randint(-10, 10)) # calculates time index where the dip should occur with random variation
        amplitude = np.random.uniform(low=-10.0, high=-7.0) # generates max amplitude it can drop
        duration_samples = int(np.random.uniform(10, 14)) # how long the drop will occur

        # find start and end of the event occurring
        start = max(0, center_index - duration_samples // 2)
        end = min(n_samples, center_index + duration_samples // 2)

        window_size = end - start
        if window_size > 0:
            x = np.linspace(-np.pi, np.pi, window_size) # evenly distributed values across window (separated by pi)
            # generates cos function to mimic the u shaped dips
            dip_shape = (1 - np.cos(x)) / 2
            abnormal_dips[start:end] += amplitude * dip_shape

    # add all the data together
    spo2_raw = base_line + noise + abnormal_dips

    # Clip to realistic (concerning) range (min 80% during crazy and insane event, max 100%)
    spo2_final = np.clip(spo2_raw, 80.0, 100.0)

    results = pd.DataFrame({'timestamp': time_h, 'spo2_percent': spo2_final})
    results.to_csv('abnormal_spo_dummy_results.csv', index=False)

    return results

def obtain_Spo(file_path, channel='spo2_percent', time_channel='timestamp'):
    try:
        # Load the CSV data
        raw = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None

    # obtain spo and time data
    output_df = raw[[time_channel, channel]].copy()
    output_df.columns = ['time', 'data'] # rename the headers

    # push into a csv
    output_df.to_csv('SPO_results.csv', index=False)

if __name__ == "__main__":
    # Generate random data if desirable
    # Comment these two lines out if using real data
    healthy_results = generate_healthy(n_samples, time_vector)
    abnormal_results = generate_abnormal_spo2(n_samples, time_vector)

    # obtain the data and store it in csv
    obtain_Spo(file_path='esp32_recorded_data.csv', channel='spo2_percent', time_channel='timestamp')