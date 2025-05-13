# Về GitHub này
Đây là repo GitHub lưu trữ mã nguồn thử nghiệm thuật toán LSTM (Long-Short Term Memory) đã được cải tiến bằng giải thuật di truyền (Genetic Algorithm, GA) phục vụ dự đoán giá chứng khoán của một mã chứng khoán cụ thể.

Dữ liệu được lấy bằng thư viên yfinance: [https://github.com/ranaroussi/yfinance/](https://github.com/ranaroussi/yfinance/)

# Yêu cầu đối với hệ thống
1. Hệ điều hành: Windows 10/11, macOS hoặc Linux, phiên bản 64-bit (x86-64 hoặc arm64)
2. Đã cài đặt các phần mềm sau:
- Visual Studio Code, đã cài extension Python, Jupyter
- Python (phiên bản 3.9 trở lên) cùng với trình quản lí gói pip

Trong trường hợp chưa có phần mềm yêu cầu, vui lòng tham khảo theo hướng dẫn ứng với hệ điều hành máy bạn sử dụng.

# Hướng dẫn chạy chương trình
1. Tải repo này về máy của bạn.
2. Mở Visual Studio Code, chọn `File -> Open Folder` và mở folder repo bạn vừa mới tải về
3. Tại thanh dọc Explorer, mở file `main.ipynb`
4. Sau khi mở `main.ipynb`, các bạn cần cấu hình môi trường để có thể chạy code Python trong Jupyter notebook.
- Nhấn `Ctrl + Shift + P` (macOS: `Super + Shift + P`). Ô nhập lệnh hiện ra (giữa bên trên màn hình), nhập từ khóa: `Interpreter`. Chọn: `Python: Select Interpreter`. - Một danh sách hiện ra, chọn `Create Virtual Environment`. Tại danh sách kế tiếp, chọn `Venv: Creates a 'vemv' ...`. Tiếp theo chọn `Python 3.x.y 64-bit` _(lưu ý: phải chọn phiên bản Python lớn hơn 3.9.0)_.
    - Nếu hiện ra thông báo: `The following environment is selected: ...` tức là thao tác trên đã thành công
- Tại góc phải màn hình, chọn `Select Kernel`. Một danh sách hiện ra, chọn `.venv (Python 3.x.y)` mà bạn đã tạo trước đó. Nếu nút `Select Kernel: ...` chuyển sang `.venv (Python 3.x.y)` tức là đã thành công.
- Cuối cùng, bạn cần cài đặt các thư viện cần thiết để code Python bên trong Jupyter notebook này có thể chạy được. Trên thanh công cụ, chọn `Terminal -> New Terminal`. Một giao diện dòng lệnh sẽ hiện ra ở bên dưới màn hình. Nhập và chạy lệnh sau:
```zsh
pip install ipykernel numpy pandas matplotlib yfinance
```
- Nếu lệnh không thông báo lỗi gì, tức là bạn đã thiết lập thành công môi trường Python cho Jupyter notebook. Giờ bạn có thể chỉnh tham số trong Jupyter notebook theo ý muốn (ví dụ, mã chứng khoán cần dự đoán) và chạy theo hướng dẫn đã được ghi kèm theo trong notebook.