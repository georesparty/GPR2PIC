import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def read_gssi_dzt_final(file_path, default_scans_per_meter):
    """
    Final version using the empirically discovered offsets from the user's hex dump.
    """
    try:
        with open(file_path, 'rb') as f:
            header_data = f.read(1024)
            if len(header_data) < 1024:
                print(f"Error: File '{os.path.basename(file_path)}' header is less than 1024 bytes.")
                return None, None

            # --- Definitive Parameter Reading ---

            # Channel Count @ offset 51 (Big-Endian) - Confirmed Correct
            num_channels = struct.unpack('>h', header_data[51:53])[0]
            print(f"File '{os.path.basename(file_path)}' identified, Channels: {num_channels}")

            # Samples per Trace @ offset 4 (Little-Endian) - Discovered from Hex Dump
            samples_per_trace = struct.unpack('<h', header_data[4:6])[0]

            # Bits per Sample @ offset 6 (Little-Endian) - Discovered from Hex Dump
            bits_per_sample = struct.unpack('<h', header_data[6:8])[0]

            # Scans per Meter @ offset 40 (Little-Endian) - We will fall back to default if zero
            scans_per_meter = struct.unpack('<f', header_data[40:44])[0]

            # --- Validation ---
            print(
                f"  - [Debug Info] Header Params: samples_per_trace={samples_per_trace}, bits_per_sample={bits_per_sample}, scans_per_meter={scans_per_meter:.3f}")

            if samples_per_trace <= 0 or bits_per_sample not in [8, 16, 32]:
                print(f"  - Error: Invalid header parameters read.")
                return None, None

            # --- Data Location and Trace Count ---
            data_start_position = 1024

            f.seek(0, os.SEEK_END)
            data_size = f.tell() - data_start_position

            bytes_per_sample = bits_per_sample // 8
            denominator = samples_per_trace * bytes_per_sample
            total_traces = data_size // denominator

            if total_traces <= 0:
                print(f"Warning: Calculated trace count is zero for '{os.path.basename(file_path)}'.")
                return None, None

            if scans_per_meter <= 0:
                print(
                    f"  - Warning: 'Scans per Meter' in file is zero. Using user-defined default: {default_scans_per_meter}")
                scans_per_meter = default_scans_per_meter

            # --- Read Data ---
            f.seek(data_start_position)
            # Data is Little-Endian, matching the format for most of the header
            data_type = '<i2' if bits_per_sample == 16 else '<i4'

            raw_data = np.fromfile(f, dtype=data_type, count=total_traces * samples_per_trace)
            radar_data_matrix = raw_data.reshape((samples_per_trace, total_traces), order='F')

            print(f"  - Data read successfully!")

            header_info = {
                "samples_per_trace": samples_per_trace,
                "bits_per_sample": bits_per_sample,
                "scans_per_meter": scans_per_meter,
                "trace_count": total_traces,
                "total_length_m": total_traces / scans_per_meter,
            }

            print(f"  - Traces: {total_traces}, Total Length: {header_info['total_length_m']:.2f} m")
            return radar_data_matrix, header_info

    except Exception as e:
        print(f"A critical error occurred while processing '{file_path}': {e}")
        return None, None


def _plot_and_save_image(data, title, output_path, dpi):
    plt.figure(figsize=(16, 8))
    vmax = np.percentile(data, 99)
    vmin = -vmax
    plt.imshow(data, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=16)
    plt.xlabel('Trace Number', fontsize=12)
    plt.ylabel('Sample Number', fontsize=12)
    plt.colorbar(label='Amplitude')
    try:
        plt.savefig(output_path, format='jpg', dpi=dpi, bbox_inches='tight')
        print(f"  -> Image saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"Error: Could not save image '{output_path}': {e}")
    plt.close()


def process_and_export_images(radar_data, header_info, base_filename, output_dir, **kwargs):
    length_threshold = kwargs.get('length_threshold_m', 100)
    window_length_traces = kwargs.get('window_traces', 1200)
    output_dpi = kwargs.get('dpi', 300)
    total_length = header_info['total_length_m']
    if total_length < length_threshold:
        print(f"  - Line length {total_length:.2f} m < {length_threshold} m. Exporting as a single image.")
        output_path = os.path.join(output_dir, f"{base_filename}_full.jpg")
        title = f'GPR Profile: {base_filename} (Length: {total_length:.2f} m)'
        _plot_and_save_image(radar_data, title, output_path, output_dpi)
    else:
        print(f"  - Line length {total_length:.2f} m >= {length_threshold} m. Splitting into windows.")
        total_traces = header_info['trace_count']
        for i, start_trace in enumerate(range(0, total_traces, window_length_traces)):
            end_trace = min(start_trace + window_length_traces, total_traces)
            if start_trace == end_trace: continue
            data_window = radar_data[:, start_trace:end_trace]
            output_path = os.path.join(output_dir, f"{base_filename}_window_{i + 1:03d}.jpg")
            title = f'GPR Profile: {base_filename} - Window {i + 1}'
            _plot_and_save_image(data_window, title, output_path, output_dpi)
    print(f"File '{base_filename}' processed successfully.")


def main():
    # --- User Configuration ---
    data_directory = '.'
    DEFAULT_SCANS_PER_METER = 200.0
    output_dpi = 300
    length_threshold_m = 100
    window_traces = 1200

    output_directory = os.path.join(data_directory, 'JPG_Output_Final')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    found_files = False
    for filename in os.listdir(data_directory):
        if filename.lower().endswith('.dzt'):
            found_files = True
            file_path = os.path.join(data_directory, filename)
            radar_data, header_info = read_gssi_dzt_final(file_path, DEFAULT_SCANS_PER_METER)

            if radar_data is not None:
                base_name = os.path.splitext(filename)[0]
                process_and_export_images(
                    radar_data, header_info, base_name, output_directory,
                    length_threshold_m=length_threshold_m,
                    window_traces=window_traces,
                    dpi=output_dpi
                )
            print("-" * 50)

    if not found_files:
        print(f"No .dzt files found in the directory: '{os.path.abspath(data_directory)}'")


if __name__ == '__main__':
    main()