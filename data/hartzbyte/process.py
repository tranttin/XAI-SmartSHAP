import json
import csv


def process_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        fieldnames = ['title', 'abstract', 'categories', 'update_date']
        count = 0

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in data:
                # Kiểm tra xem item có phải là dictionary không, nếu là chuỗi thì bỏ qua hoặc xử lý khác
                if not isinstance(item, dict):
                    continue

                # 1. Tự tạo Title
                raw_input = item.get('input', '')
                words = str(raw_input).split()
                title = " ".join(words[:15]) + "..." if len(words) > 15 else raw_input

                # 2. Xử lý Abstract từ output
                output_data = item.get('output', {})
                abstract = ""

                if isinstance(output_data, dict):
                    summary_list = output_data.get('summary', [])
                    if isinstance(summary_list, list):
                        abstract = " ".join(summary_list)
                    else:
                        abstract = str(summary_list)
                else:
                    # Nếu output là string (như trường hợp 3 bạn gửi)
                    abstract = str(output_data)

                # 3. Xử lý Categories từ output.methods
                categories = ""
                if isinstance(output_data, dict):
                    methods_list = output_data.get('methods', [])
                    if isinstance(methods_list, list):
                        categories = ", ".join(methods_list)
                    else:
                        categories = str(methods_list)

                # 4. Update Date
                update_date = item.get('update_date', '')

                writer.writerow({
                    'title': title,
                    'abstract': abstract,
                    'categories': categories,
                    'update_date': update_date
                })
                count += 1

        print("-" * 30)
        print(f"Hoàn tất! Đã xử lý thành công {count} mục.")
        print("-" * 30)

    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    process_data('source.txt', 'papers.csv')