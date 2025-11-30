import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import arabic_reshaper
from bidi.algorithm import get_display
import seaborn as sns
import colorsys

# Set up Persian font
# FONT_PATH = 'fonts/Vazirmatn-Regular.ttf'  # Adjust to your font path
FONT_PATH = '/usr/share/fonts/truetype/fonts/ttf/Vazirmatn-Regular.ttf'  # Adjust to your font path
if os.path.exists(FONT_PATH):
    persian_font = fm.FontProperties(fname=FONT_PATH)
else:
    print("Warning: Persian font not found. Using default font.")
    persian_font = fm.FontProperties()  # Fallback to default font

def load_scores(eval_dir, findings=True):
    scores_by_folder = {}
    # this_one = {}
    for i, folder in enumerate(os.listdir(eval_dir)):
        # this_one[folder] = f"آزمایش {i}"
        folder_path = os.path.join(eval_dir, folder)
        if os.path.isdir(folder_path):
            if 'chexpert' in folder_path or 'whole_l2_include_score_2' in folder_path:
                continue
            if findings:
                json_path = os.path.join(folder_path, 'findings_scores.json')
            else:
                json_path = os.path.join(folder_path, 'impression_scores.json')

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    scores = json.load(f)
                    scores_by_folder[folder] = scores
    # return scores_by_folder, this_one
    return scores_by_folder


# Mapping of experiment folder names to Persian names
# persian_mapping = {
#     'baseline': 'آزمایش 1',
#     'random_retrieve_include_score': 'آزمایش 2',
#     # 'chexpert': 'آزمایش 2',
#     'whole_cos_include_score': 'آزمایش 3',
#     'whole_cos_no_score': 'آزمایش 4',
#     # 'whole_l2_include_score_2': 'آزمایش 1',
#     'whole_l2_include_score': 'آزمایش 5',
#     'whole_l2_no_score': 'آزمایش 6',
#     'whole_hamming_include_score': 'آزمایش 7',
#     'whole_hamming_no_score': 'آزمایش 8',
#     'whole_jaccard_include_score': 'آزمایش 9',
#     'whole_jaccard_no_score': 'آزمایش 10',
#     'whole_negative_hamming_include_score': 'آزمایش 11',
#     'whole_negative_jaccard_include_score': 'آزمایش 12',
#     'partial_include_score_simple': 'آزمایش 13',
#     'partial_no_score_simple': 'آزمایش 14',
#     'partial_include_score_related': 'آزمایش 15',
#     'partial_no_score_related': 'آزمایش 16'
# }

number_mapping = {
    'baseline': '1',
    'random_retrieve_include_score': '2',
    'whole_cos_include_score': '3',
    'whole_cos_no_score': '4',
    'whole_l2_include_score': '5',
    'whole_l2_no_score': '6',
    'whole_hamming_include_score': '7',
    'whole_hamming_no_score': '8',
    'whole_jaccard_include_score': '9',
    'whole_jaccard_no_score': '10',
    'whole_negative_hamming_include_score': '11',
    'whole_negative_jaccard_include_score': '12',
    'partial_include_score_simple': '13',
    'partial_no_score_simple': '14',
    'partial_include_score_related': '15',
    'partial_no_score_related': '16'
}

persian_mapping = {
    'baseline': 'پایه',
    'random_retrieve_include_score': 'تصادفی',
    'whole_cos_include_score': 'کامل / کسینوسی / شامل',
    'whole_cos_no_score': 'کامل / کسینوسی / فاقد',
    'whole_l2_include_score': 'کامل / اقلیدسی / شامل',
    'whole_l2_no_score': 'کامل / اقلیدسی / فاقد',
    'whole_hamming_include_score': 'کامل / همینگ / شامل',
    'whole_hamming_no_score': 'کامل / همینگ / فاقد',
    'whole_jaccard_include_score': 'کامل / جاکارد / شامل',
    'whole_jaccard_no_score': 'کامل / جاکارد / فاقد',
    'whole_negative_hamming_include_score': 'کامل / همینگ / شامل / با نمونه منفی',
    'whole_negative_jaccard_include_score': 'کامل / جاکارد / شامل / با نمونه منفی',
    'partial_include_score_simple': 'جزئی / ساده / شامل',
    'partial_no_score_simple': 'جزئی / ساده / فاقد',
    'partial_include_score_related': 'جزئی / مرتبط / شامل',
    'partial_no_score_related': 'جزئی / مرتبط / فاقد'
}

key_mapping = {'Average_BertScore': 'Average BertScore',
               'Average_Bleu': 'Average Bleu',
               'Average_Meteor': 'Average Meteor',
               'Average_Rouge': 'Average Rouge',
               'chexbert-5_macro avg_f1-score': 'Chexbert-5 Macro-Avg-F1-score',
               'chexbert-5_micro avg_f1-score': 'Chexbert-5 Micro-Avg-F1-score',
               'chexbert-all_macro avg_f1-score': 'Chexbert-all Macro-Avg-F1-score',
               'chexbert-all_micro avg_f1-score': 'Chexbert-all Micro-Avg-F1-score',
               'radgraph_complete': 'Radgraph-Complete',
               'radgraph_partial': 'Radgraph-Partial',
               'radgraph_simple': 'Radgraph-Simple'}


# Define pairs based on table (order: base, full pairs, partial pairs)
# Each pair gets a rainbow hue with light/dark variants
pairs = [
    # Base "pair" (rows 1-2)
    ['baseline', 'random_retrieve_include_score'],

    # Full pairs (rows 3-12)
    ['whole_cos_include_score', 'whole_cos_no_score'],  # Cosine
    ['whole_l2_include_score', 'whole_l2_no_score'],  # L2
    ['whole_hamming_include_score', 'whole_hamming_no_score'],  # Hamming
    ['whole_jaccard_include_score', 'whole_jaccard_no_score'],  # Jaccard
    ['whole_negative_hamming_include_score', 'whole_negative_jaccard_include_score'],  # Negative

    # Partial pairs (rows 13-16)
    ['partial_include_score_simple', 'partial_no_score_simple'],  # Simple
    ['partial_include_score_related', 'partial_no_score_related']  # Related
]

# Group headers for legend (optional)
group_headers = [
    'Base Experiments:',
    'Full Retrievals:',
    'Partial Retrievals:'
]

# Function to reshape Persian text for correct RTL rendering
def reshape_persian_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

def get_persian_name(folder):
    # return persian_mapping.get(folder, folder)  # Fallback to original name if not mapped
    return reshape_persian_text(persian_mapping.get(folder, folder))


# Function to calculate dynamic y-axis limits
def calculate_y_limits(scores):
    if not scores:
        return 0.0, 1.0  # Default range if no scores

    min_score = min(scores)
    max_score = max(scores)

    # Handle case where all scores are the same
    if min_score == max_score:
        if min_score == 0:
            return 0.0, 0.1
        else:
            range_size = abs(max_score) * 0.1  # 10% of value
            return max(0, min_score - range_size), max_score + range_size

    # Calculate range and add 10% padding
    range_size = max_score - min_score
    padding = range_size * 0.1  # 10% padding
    y_min = max(0, min_score - padding)  # Ensure non-negative if appropriate
    y_max = max_score + padding

    # Round to nice numbers
    magnitude = 10 ** np.floor(np.log10(range_size))
    y_min = np.floor(y_min / magnitude) * magnitude
    y_max = np.ceil(y_max / magnitude) * magnitude

    return y_min, y_max


def light_dark_from_hue(hue, light_factor=0.8, dark_factor=0.4):
    """Generate light and dark variants from a hue (0-1)"""
    # Light: high value, medium saturation
    light_rgb = colorsys.hsv_to_rgb(hue, 0.6, light_factor)
    # Dark: low value, high saturation
    dark_rgb = colorsys.hsv_to_rgb(hue, 0.9, dark_factor)
    return light_rgb, dark_rgb


def plot_bar_chart_per_metric(scores_by_folder, output_dir='charts', findings=True):
    if not scores_by_folder:
        print("No scores found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    sample_scores = next(iter(scores_by_folder.values()))
    score_keys = list(sample_scores.keys())

    # Build ordered_folders and legend from pairs (no headers, sequential)
    ordered_folders = []
    legend_labels = []
    exp_num = 1  # 1-based numbering

    for pair in pairs:
        for folder in pair:
            if folder in scores_by_folder:
                ordered_folders.append(folder)
                # Sequential numbered labels only
                legend_labels.append(reshape_persian_text(f"({exp_num}) {persian_mapping.get(folder, folder)}"))
                exp_num += 1

    # Add ungrouped at end
    all_folders = set(scores_by_folder.keys())
    grouped_folders = set(ordered_folders)
    ungrouped = list(all_folders - grouped_folders)
    for folder in ungrouped:
        ordered_folders.append(folder)
        legend_labels.append(reshape_persian_text(f"({exp_num}) {persian_mapping.get(folder, folder)}"))
        exp_num += 1

    x_labels = [str(i + 1) for i in range(len(ordered_folders))]  # Numbers 1 to N

    # Custom rainbow hues (avoid brownish orange; 8 hues for pairs)
    num_pairs = len(pairs)
    # custom_hues = [0.0, 0.02, 0.15, 0.25, 0.35, 0.45, 0.55,
    #                0.65]  # Red, pink-red, yellow-orange, yellow, lime, green, cyan, blue
    rainbow_hues = np.linspace(0, 0.8, num_pairs)  # HSV hues for rainbow (avoid full cycle for better spread)
    pair_colors = []  # List of (light, dark) for each pair

    # for hue in custom_hues[:num_pairs]:
    for hue in rainbow_hues[:num_pairs]:
        light, dark = light_dark_from_hue(hue)
        pair_colors.append((light, dark))

    # Assign colors to folders (light for first in pair, dark for second)
    colors = []
    pair_idx = 0
    for pair in pairs:
        light_color, dark_color = pair_colors[pair_idx % len(pair_colors)]
        for i, folder in enumerate(pair):
            if folder in scores_by_folder:
                if i == 0:  # First in pair: light
                    colors.append(plt.cm.colors.to_hex(light_color))
                else:  # Second: dark
                    colors.append(plt.cm.colors.to_hex(dark_color))
        pair_idx += 1

    # Fallback for ungrouped
    ungrouped_count = len(ordered_folders) - len(colors)
    if ungrouped_count > 0:
        fallback_colors = sns.color_palette('hsv', ungrouped_count)
        colors.extend([plt.cm.colors.to_hex(c) for c in fallback_colors])

    for key in score_keys:
        plt.figure(figsize=(16, 6))  # Width for legend
        scores = [scores_by_folder[folder].get(key, 0) for folder in ordered_folders]

        # Calculate dynamic y-axis limits
        y_min, y_max = calculate_y_limits(scores)

        bars = plt.bar(x_labels, scores, color=colors)

        # Add scores on bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{score:.3f}',
                ha='center',
                va='bottom',
                fontproperties=persian_font,
                fontsize=16
            )

        # X-axis: Numbers
        mapped_key = key_mapping[key]
        if findings:
            chart_title = reshape_persian_text(f'مقایسه امتیاز {mapped_key} در یافته‌ها (Findings)')
            save_name = 'findings'
        else:
            chart_title = reshape_persian_text(f'مقایسه امتیاز {mapped_key} در جمع‌بندی (Impression)')
            save_name = 'impressions'

        plt.xlabel(reshape_persian_text('شماره آزمایش'), fontproperties=persian_font, fontsize=18)
        # plt.ylabel(reshape_persian_text(f'امتیاز {mapped_key}'), fontproperties=persian_font)
        plt.ylabel(mapped_key, fontproperties=persian_font, fontsize=18)
        # plt.title(reshape_persian_text(f'مقایسه امتیاز {mapped_key} در آزمایش‌ها'), fontproperties=persian_font)
        plt.title(chart_title, fontproperties=persian_font,fontsize=18)
        plt.xticks(rotation=0)  # No rotation needed for numbers
        plt.ylim(y_min, y_max)

        # Legend: Sequential numbered names only, vertically centered
        legend_prop = persian_font.copy()
        legend_prop.set_size(15)
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(legend_labels))]
        # plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
        #            ncol=1, fontsize='xx-large', title=reshape_persian_text('آزمایش‌ها'),
        #            title_fontproperties=persian_font, prop=persian_font)
        plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
                   ncol=1, prop=legend_prop,
                   title_fontproperties=persian_font)

        plt.tight_layout()
        # plt.show()
        # Save as PDF for selectable text
        plt.savefig(os.path.join(output_dir, f'{save_name}_{key}_bar_chart.pdf'), format='pdf', bbox_inches='tight')
        plt.close()
        # break

def main():
    eval_dir = 'results'  # Path to eval folder
    # findings
    scores = load_scores(eval_dir, findings=True)
    plot_bar_chart_per_metric(scores, findings=True)
    # impression
    scores = load_scores(eval_dir, findings=False)
    plot_bar_chart_per_metric(scores, findings=False)



def plot_bar_charts_txr(thresholds, mean_scores, max_scores, micro=True):
    # Set up bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(thresholds))

# Plot bars for Mean and Max
    bars1 = ax.bar(x - bar_width/2, mean_scores, bar_width, label=get_persian_name("تجمیع به روش میانگین"), color="skyblue")
    bars2 = ax.bar(x + bar_width/2, max_scores, bar_width, label=get_persian_name("تجمیع به روش بیشینه"), color="salmon")

    # Highlight maximum scores
    max_mean_idx = mean_scores.index(max(mean_scores))  # Index of max Mean score (84.10 at 0.65)
    max_max_idx = max_scores.index(max(max_scores))    # Index of max Max score (82.98 at 0.7)
    ax.text(x[max_mean_idx] - bar_width/2, mean_scores[max_mean_idx] + 0.5, f"{mean_scores[max_mean_idx]}", ha="center", fontproperties=persian_font)
    ax.text(x[max_max_idx] + bar_width/2, max_scores[max_max_idx] + 0.5, f"{max_scores[max_max_idx]}", ha="center", fontproperties=persian_font)

    # Customize chart
    ax.set_xlabel(get_persian_name("آستانه"), fontproperties=persian_font)
    if micro:
        ax.set_ylabel(get_persian_name("امتیاز micro-F1"), fontproperties=persian_font)
        ax.set_title(get_persian_name("مقایسه‌ی micro-F1 مدل TXR با روش‌های تجمیع میانگین و بیشینه در مجموعه‌ی اعتبارسنجی"), fontproperties=persian_font)
    else:
        ax.set_ylabel(get_persian_name("امتیاز macro-F1"), fontproperties=persian_font)
        ax.set_title(get_persian_name("مقایسه‌ی macro-F1 مدل TXR با روش‌های تجمیع میانگین و بیشینه در مجموعه‌ی اعتبارسنجی"), fontproperties=persian_font)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds], fontproperties=persian_font)
    ax.legend(prop=persian_font)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.ylim(30, 90)

    plt.show()
    # Save the chart
    # plt.savefig("micro_f1_comparison.png", dpi=300, bbox_inches="tight")
    # Save as PDF for selectable text
    # if micro:
    #     plt.savefig(os.path.join('./results/charts/', f'micro_f1_val.pdf'), dpi=300, format='pdf', bbox_inches='tight')
    # else:
    #     plt.savefig(os.path.join('./results/charts/', f'macro_f1_val.pdf'), dpi=300, format='pdf', bbox_inches='tight')
    plt.close()


def metrics_table_str():
    jump_list = [
        'chexbert-5_macro avg_f1-score',
        'chexbert-all_macro avg_f1-score',
        'chexbert-all_micro avg_f1-score',
        'radgraph_complete',
        'radgraph_simple'
    ]
    eval_dir = 'results'  # Path to eval folder
    scores = load_scores(eval_dir, findings=True)
    # scores = load_scores(eval_dir, findings=False)
    for exp, metrics in scores.items():
    #     آزمایش پایه & $0.444$ & $0.016$ & $0.185$ & $0.137$ & $0.505$ & $0.103
        if exp not in persian_mapping.keys():
            continue
        # mapped_exp = persian_mapping[exp]
        mapped_exp = number_mapping[exp]
        row_str = mapped_exp + ' & '
        for key, m in metrics.items():
            if key in jump_list:
                continue
            row_str += f'${m:.3f}$ & '
        # print(row_str)
        print(row_str[:-2] + '\\\\')
        # break

    # print()

def datasets_chart():

    datasets = ["CheXpert", "CheXpert Plus", "MIMIC-CXR", "MIMIC-CXR-JPG", "IU-XRAY", "PadChest", "NIH ChestX-ray14"]
    images = [224316, 223228, 377095, 377095, 7470, 160000, 112120]
    patients = [65240, 64725, 65379, 65379, 3955, 67000, 30805]

    # Set up the bar chart
    x = np.arange(len(datasets))  # Positions for bars
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars for images and patients
    ax.bar(x - width / 2, images, width, label=reshape_persian_text('تعداد تصاویر'), color='#36A2EB')
    # Filter out None values for patients
    patients_valid = [p if p is not None else 0 for p in patients]
    ax.bar(x + width / 2, patients_valid, width, label=reshape_persian_text('تعداد بیماران'), color='#FF6384')
    ax.legend(prop=persian_font)

    # Customize the chart
    # ax.set_xlabel(reshape_persian_text('مجموعه‌دادگان'), fontproperties=persian_font)
    ax.set_ylabel(reshape_persian_text('تعداد'), fontproperties=persian_font)
    ax.set_title(reshape_persian_text('مقایسه مجموعه‌دادگان رادیولوژی'), fontproperties=persian_font)
    ax.set_xticks(x)
    # ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_xticklabels(datasets, ha='center')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    # Save the chart as a PNG
    # plt.savefig('radiology_datasets_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # datasets_chart()
    metrics_table_str()

    # main()
    # micro-f1
    # thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # mean_scores = [61.17, 64.35, 66.55, 80.47, 83.66, 84.10, 83.44, 82.34, 80.81, 79.35, 78.04, 76.86]
    # max_scores = [42.48, 44.32, 45.98, 74.73, 80.64, 82.59, 82.98, 82.45, 81.32, 79.71, 78.25, 76.92]
    # plot_bar_charts_txr(thresholds, mean_scores, max_scores)

    # macro-f1
    # thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # mean_scores = [60.02, 62.73, 64.55, 75.98, 77.84, 76.70, 73.75, 69.66, 64.33, 58.28, 52.14, 45.93]
    # max_scores = [42.42, 44.31, 45.98, 71.01, 75.31, 75.81, 74.19, 71.19, 66.49, 59.93, 53.21, 46.27]
    # plot_bar_charts_txr(thresholds, mean_scores, max_scores, micro=False)
