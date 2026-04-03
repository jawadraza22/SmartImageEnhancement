import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==========================================
# GLOBAL VARIABLES
# ==========================================
original = None
gray = None
processed = None
analysis_text = None

# ==========================================
# GRADIENT BACKGROUND
# ==========================================
def draw_gradient(event=None):
    canvas.delete("all")
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    for i in range(height):
        r = int(100 + (i / height) * 100)
        g = int(150 + (i / height) * 80)
        b = 255
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, width, i, fill=color)

# ==========================================
# DISPLAY IMAGE
# ==========================================
def display(img, panel, max_size=(320, 320)):
    if img is None:
        return
    # Convert grayscale to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    # Resize while keeping aspect ratio
    w, h = img.size
    max_w, max_h = max_size
    ratio = min(max_w/w, max_h/h)
    img = img.resize((int(w*ratio), int(h*ratio)))
    
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# ==========================================
# LOAD IMAGE
# ==========================================
def load_image():
    global original, gray, processed, analysis_text
    path = filedialog.askopenfilename()
    if not path:
        return
    original = cv2.imread(path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed = gray.copy()
    update_analysis()
    display(original, panelA)
    display(processed, panelB)

# ==========================================
# IMAGE ANALYSIS PANEL
# ==========================================
def update_analysis():
    if gray is None:
        return
    info = f"Resolution: {gray.shape[1]}x{gray.shape[0]}\n"
    info += f"Data type: {gray.dtype}\n"
    info += f"Sample matrix (5x5 top-left):\n{gray[:5,:5]}\n"
    analysis_text.config(state=NORMAL)
    analysis_text.delete("1.0", END)
    analysis_text.insert(END, info)
    analysis_text.config(state=DISABLED)

# ==========================================
# APPLY PIPELINE
# ==========================================
def apply_pipeline():
    global processed
    if gray is None:
        messagebox.showerror("Error", "Load image first!")
        return
    processed = gray.copy()

    # Intensity transformations
    if negative_var.get():
        processed = 255 - processed
    if log_var.get():
        c = 255 / np.log(1 + np.max(processed))
        processed = c * np.log(1 + processed)
        processed = np.uint8(processed)
    if gamma_var.get():
        g = float(gamma_value.get())
        norm = processed / 255.0
        processed = np.power(norm, g)
        processed = np.uint8(processed * 255)

    # Geometric transformations
    if rotate_var.get():
        angle = int(angle_var.get())
        h, w = processed.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        processed = cv2.warpAffine(processed, M, (w, h))
        if inverse_rotate_var.get():
            M_inv = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1)
            processed = cv2.warpAffine(processed, M_inv, (w, h))

    if translate_var.get():
        tx = int(tx_var.get())
        ty = int(ty_var.get())
        M = np.float32([[1,0,tx],[0,1,ty]])
        processed = cv2.warpAffine(processed, M, (processed.shape[1], processed.shape[0]))
        if inverse_translate_var.get():
            M_inv = np.float32([[1,0,-tx],[0,1,-ty]])
            processed = cv2.warpAffine(processed, M_inv, (processed.shape[1], processed.shape[0]))

    if shear_var.get():
        shx = float(shx_var.get())
        shy = float(shy_var.get())
        M = np.float32([[1, shx, 0],[shy,1,0]])
        processed = cv2.warpAffine(processed, M, (processed.shape[1], processed.shape[0]))
        if inverse_shear_var.get():
            M_inv = np.float32([[1,-shx,0],[-shy,1,0]])
            processed = cv2.warpAffine(processed, M_inv, (processed.shape[1], processed.shape[0]))

    # Sampling & Quantization
    if sample_var.get():
        scale = float(scale_var.get())
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if quant_var.get():
        bits = int(bit_var.get())
        levels = 2 ** bits
        step = 256 // levels
        processed = (processed // step) * step

    # Histogram equalization
    if hist_var.get():
        processed = cv2.equalizeHist(processed)

    display(processed, panelB)
    update_analysis()

# ==========================================
# SHOW HISTOGRAM
# ==========================================
def show_histogram():
    if processed is None:
        return
    plt.figure("Histogram")
    plt.hist(processed.ravel(), bins=256, color='gray')
    plt.title("Histogram")
    plt.show()

# ==========================================
# SAVE IMAGE
# ==========================================
def save_image():
    if processed is None:
        return
    cv2.imwrite("enhanced_output.jpg", processed)
    messagebox.showinfo("Saved", "Image saved as 'enhanced_output.jpg'!")

# ==========================================
# MAIN WINDOW
# ==========================================
root = Tk()
root.title("Smart Image Enhancement System")
root.state("zoomed")

canvas = Canvas(root)
canvas.pack(fill="both", expand=True)
canvas.bind("<Configure>", draw_gradient)

# ==========================================
# CONTROL FRAME (LEFT PANEL)
# ==========================================
frame = Frame(canvas, bg="#e0f7fa", bd=2, relief=RIDGE)
frame.place(relx=0.02, rely=0.05, anchor="nw")

# Variables
rotate_var = IntVar()
inverse_rotate_var = IntVar()
translate_var = IntVar()
inverse_translate_var = IntVar()
shear_var = IntVar()
inverse_shear_var = IntVar()
sample_var = IntVar()
quant_var = IntVar()
gamma_var = IntVar()
negative_var = IntVar()
log_var = IntVar()
hist_var = IntVar()

angle_var = StringVar(value="45")
tx_var = StringVar(value="50")
ty_var = StringVar(value="50")
shx_var = StringVar(value="0.2")
shy_var = StringVar(value="0.2")
scale_var = StringVar(value="1")
bit_var = StringVar(value="4")
gamma_value = StringVar(value="0.5")

# Styles
btn_main = {"bg": "#00796B","fg":"white","font":("Arial",11,"bold"),"bd":0,"width":18}
btn_secondary = {"bg": "#0288D1","fg":"white","font":("Arial",11,"bold"),"bd":0,"width":18}
chk_style = {"bg": "#e0f7fa","font":("Arial",11),"activebackground": "#b2ebf2"}

# Controls
Button(frame, text="Load Image", command=load_image, **btn_main).grid(row=0, column=0, pady=5)
Checkbutton(frame, text="Negative", variable=negative_var, **chk_style).grid(row=1,column=0, sticky="w")
Checkbutton(frame, text="Log", variable=log_var, **chk_style).grid(row=2,column=0, sticky="w")
Checkbutton(frame, text="Gamma", variable=gamma_var, **chk_style).grid(row=3,column=0, sticky="w")
OptionMenu(frame, gamma_value, "0.5","1.5").grid(row=3,column=1)
Checkbutton(frame, text="Rotation", variable=rotate_var, **chk_style).grid(row=4,column=0, sticky="w")
OptionMenu(frame, angle_var,"30","45","60","90","120","150","180").grid(row=4,column=1)
Checkbutton(frame, text="Inverse Rotation", variable=inverse_rotate_var, **chk_style).grid(row=4,column=2)
Checkbutton(frame, text="Translation", variable=translate_var, **chk_style).grid(row=5,column=0, sticky="w")
Entry(frame,textvariable=tx_var,width=5).grid(row=5,column=1)
Entry(frame,textvariable=ty_var,width=5).grid(row=5,column=2)
Checkbutton(frame,text="Inverse Translation", variable=inverse_translate_var, **chk_style).grid(row=5,column=3, sticky="w")
Checkbutton(frame, text="Shearing", variable=shear_var, **chk_style).grid(row=6,column=0, sticky="w")
Entry(frame,textvariable=shx_var,width=5).grid(row=6,column=1)
Entry(frame,textvariable=shy_var,width=5).grid(row=6,column=2)
Checkbutton(frame,text="Inverse Shear", variable=inverse_shear_var, **chk_style).grid(row=6,column=3, sticky="w")
Checkbutton(frame,text="Sampling", variable=sample_var, **chk_style).grid(row=7,column=0, sticky="w")
OptionMenu(frame, scale_var,"0.25","0.5","1","1.5","2").grid(row=7,column=1)
Checkbutton(frame,text="Quantization", variable=quant_var, **chk_style).grid(row=8,column=0, sticky="w")
OptionMenu(frame, bit_var,"8","4","2").grid(row=8,column=1)
Checkbutton(frame,text="Histogram Equalization", variable=hist_var, **chk_style).grid(row=9,column=0, sticky="w")
Button(frame,text="Apply Pipeline", command=apply_pipeline, **btn_main).grid(row=10,column=0,pady=5)
Button(frame,text="Show Histogram", command=show_histogram, **btn_secondary).grid(row=10,column=1,pady=5)
Button(frame,text="Save Image", command=save_image, **btn_main).grid(row=11,column=0,pady=5)

# ==========================================
# IMAGE PANELS (BOTTOM-RIGHT)
# ==========================================
panelA = Label(canvas)
panelA.place(relx=0.98, rely=0.98, anchor="se")  # original image
panelB = Label(canvas)
panelB.place(relx=0.75, rely=0.98, anchor="se")  # processed image

# ==========================================
# ANALYSIS PANEL
# ==========================================
analysis_text = Text(canvas, width=40, height=10, font=("Arial",11))
analysis_text.place(relx=0.02, rely=0.75)
analysis_text.config(state=DISABLED)

root.mainloop()