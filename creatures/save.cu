#include "creatures/creatures.cuh"
#include <fstream>
#include <iostream>



#define SAVE_NAURON_VALUES false
#define SAVE_NETWORK_WEIGHTS false

void Creatures::Save(int first_newborn_index, int t) {
    int new_creatures_count = count - first_newborn_index;

    // old creatures only
    cudaMemcpyAsync(h_pinned->chosen_action, h_data->chosen_action, sizeof(int8_t) * first_newborn_index, cudaMemcpyDeviceToHost, transfer_stream);
#if SAVE_NAURON_VALUES
    for (int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
        cudaMemcpyAsync(
            h_pinned->input_layer_values + input_index * first_newborn_index,
            h_data->input_layer_values + get_input_layer_value_idx(0, input_index),
            sizeof(__nv_fp8_e4m3) * first_newborn_index,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }

    for(int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        cudaMemcpyAsync(
            h_pinned->hidden_layer_values+ hidden_index * first_newborn_index,
            h_data->hidden_layer_values + get_hidden_layer_value_idx(0, hidden_index),
            sizeof(__nv_fp8_e4m3) * first_newborn_index,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }

    for(int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
        cudaMemcpyAsync(
            h_pinned->output_layer_values + output_index * first_newborn_index,
            h_data->output_layer_values + get_output_layer_value_idx(0, output_index),
            sizeof(float) * first_newborn_index,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }
#endif

    // all creatures
    cudaMemcpyAsync(h_pinned->x, h_data->x, sizeof(unsigned int) * count, cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_pinned->y, h_data->y, sizeof(unsigned int) * count, cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_pinned->energy, h_data->energy, sizeof(float) * count, cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_pinned->water, h_data->water, sizeof(float) * count, cudaMemcpyDeviceToHost, transfer_stream);
    cudaMemcpyAsync(h_pinned->ids, h_data->ids, sizeof(long long) * count, cudaMemcpyDeviceToHost, transfer_stream);

    // new creatures only
    for (int sensor_index = 0; sensor_index < MILIEU_SENSORS_N; sensor_index++) {
        cudaMemcpyAsync(
            h_pinned->sensor_x + sensor_index * new_creatures_count,
            h_data->sensor_x + get_sensor_idx(first_newborn_index, sensor_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
        cudaMemcpyAsync(
            h_pinned->sensor_y + sensor_index * new_creatures_count,
            h_data->sensor_y + get_sensor_idx(first_newborn_index, sensor_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
        cudaMemcpyAsync(
            h_pinned->sensor_type + sensor_index * new_creatures_count,
            h_data->sensor_type + get_sensor_idx(first_newborn_index, sensor_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }

#if SAVE_NETWORK_WEIGHTS
    for (int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        for (int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
            cudaMemcpyAsync(
                h_pinned->first_matrix + (hidden_index * INPUT_NEURONS_N + input_index) * new_creatures_count,
                h_data->first_matrix + get_first_matrix_idx(first_newborn_index, hidden_index, input_index),
                sizeof(__nv_fp8_e4m3) * new_creatures_count,
                cudaMemcpyDeviceToHost,
                transfer_stream
            );
        }
        for (int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
            cudaMemcpyAsync(
                h_pinned->second_matrix + (output_index * HIDDEN_NEURONS_N + hidden_index) * new_creatures_count,
                h_data->second_matrix + get_second_matrix_idx(first_newborn_index, output_index, hidden_index),
                sizeof(__nv_fp8_e4m3) * new_creatures_count,
                cudaMemcpyDeviceToHost,
                transfer_stream
            );
        }
        cudaMemcpyAsync(
            h_pinned->bias + hidden_index * new_creatures_count,
            h_data->bias + get_hidden_layer_bias_idx(first_newborn_index, hidden_index),
            sizeof(__nv_fp8_e4m3) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }
#endif

    for (int action_index = 0; action_index < ACTION_TYPES_N; action_index++) {
        cudaMemcpyAsync(
            h_pinned->action_x + action_index * new_creatures_count,
            h_data->action_x + get_action_idx(first_newborn_index, action_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
        cudaMemcpyAsync(
            h_pinned->action_y + action_index * new_creatures_count,
            h_data->action_y + get_action_idx(first_newborn_index, action_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
        cudaMemcpyAsync(
            h_pinned->action_type + action_index * new_creatures_count,
            h_data->action_type + get_action_idx(first_newborn_index, action_index),
            sizeof(int8_t) * new_creatures_count,
            cudaMemcpyDeviceToHost,
            transfer_stream
        );
    }

    cudaStreamSynchronize(transfer_stream);

    if (!save_stream.is_open()) return;

    // 1. Write the tick header (metadata needed for reading later)
    save_stream.write(reinterpret_cast<const char*>(&t), sizeof(int));
    save_stream.write(reinterpret_cast<const char*>(&count), sizeof(int));
    save_stream.write(reinterpret_cast<const char*>(&first_newborn_index), sizeof(int));
    save_stream.write(reinterpret_cast<const char*>(&new_creatures_count), sizeof(int));

    // 2. Write the data blocks directly from host pinned memory

    // Old creatures data
    save_stream.write(reinterpret_cast<const char*>(h_pinned->chosen_action), sizeof(int8_t) * first_newborn_index);

#if SAVE_NAURON_VALUES
    for (int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->input_layer_values + input_index * first_newborn_index), 
            sizeof(__nv_fp8_e4m3) * first_newborn_index
        );
    }

    for (int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->hidden_layer_values + hidden_index * first_newborn_index), 
            sizeof(__nv_fp8_e4m3) * first_newborn_index
        );
    }

    for (int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->output_layer_values + output_index * first_newborn_index), 
            sizeof(float) * first_newborn_index
        );
    }
#endif

    // All creatures data
    save_stream.write(reinterpret_cast<const char*>(h_pinned->x), sizeof(unsigned int) * count);
    save_stream.write(reinterpret_cast<const char*>(h_pinned->y), sizeof(unsigned int) * count);
    save_stream.write(reinterpret_cast<const char*>(h_pinned->energy), sizeof(float) * count);
    save_stream.write(reinterpret_cast<const char*>(h_pinned->water), sizeof(float) * count);
    save_stream.write(reinterpret_cast<const char*>(h_pinned->ids), sizeof(long long) * count);

    // New creatures data
    for (int sensor_index = 0; sensor_index < MILIEU_SENSORS_N; sensor_index++) {
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->sensor_x + sensor_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->sensor_y + sensor_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->sensor_type + sensor_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
    }

#if SAVE_NETWORK_WEIGHTS
    for (int hidden_index = 0; hidden_index < HIDDEN_NEURONS_N; hidden_index++) {
        for (int input_index = 0; input_index < INPUT_NEURONS_N; input_index++) {
            save_stream.write(
                reinterpret_cast<const char*>(h_pinned->first_matrix + (hidden_index * INPUT_NEURONS_N + input_index) * new_creatures_count),
                sizeof(__nv_fp8_e4m3) * new_creatures_count
            );
        }
        for (int output_index = 0; output_index < OUTPUT_NEURONS_N; output_index++) {
            save_stream.write(
                reinterpret_cast<const char*>(h_pinned->second_matrix + (output_index * HIDDEN_NEURONS_N + hidden_index) * new_creatures_count),
                sizeof(__nv_fp8_e4m3) * new_creatures_count
            );
        }
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->bias + hidden_index * new_creatures_count),
            sizeof(__nv_fp8_e4m3) * new_creatures_count
        );
    }
#endif

    for (int action_index = 0; action_index < ACTION_TYPES_N; action_index++) {
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->action_x + action_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->action_y + action_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
        save_stream.write(
            reinterpret_cast<const char*>(h_pinned->action_type + action_index * new_creatures_count), 
            sizeof(int8_t) * new_creatures_count
        );
    }
}