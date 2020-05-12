import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance
from datetime import datetime
import os
import sys
import platform

class MainWindow():

    def __init__(self, main, result, image_path, image_folders, n):

        self.main = main
        self.result = result
        self.mainFrame = tk.Frame(self.main, bg='yellow')
        self.mainFrame.grid(row=0, column=0, sticky='news')
        self.mainFrame.grid_rowconfigure(0, weight=1)
        self.mainFrame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self.mainFrame, bg='red')
        self.canvas.grid(row=0, column=0, sticky="news")
        self.vsb = tk.Scrollbar(self.mainFrame, orient="vertical", command=self.canvas.yview)
        self.vsb.grid(row=0, column=1, sticky="ns")
        self.hsb = tk.Scrollbar(self.mainFrame, orient="horizontal", command=self.canvas.xview)
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.button = tk.Button(self.mainFrame, text="Confirm", command=self.onButtonClick)
        self.button.grid(row=2, column=0)

        self.canvas.configure(yscrollcommand=self.vsb.set,
                              xscrollcommand=self.hsb.set)
        self.frame_image = tk.Frame(self.canvas, bg='blue')
        self.frame_image.grid(row=0, column=0, sticky='news')

        self.result = result
        self.image_path = image_path
        self.image_folders = image_folders[:n]
        self.images = []
        self.selected = []
        self.flagged = []
        for i, elem in enumerate(self.image_folders):
            self.images.append([k for k in sorted(os.listdir(os.path.join(os.path.dirname(__file__), image_path, elem)),
                                                  key=lambda s: datetime.strptime(s[:12], '%H-%M-%S-%f'))
                                if 'rgb' in k])
            self.selected.append([1 for k in os.listdir(os.path.join(os.path.dirname(__file__), image_path, elem))
                                  if 'rgb' in k])

        for i, elem in enumerate(self.image_folders):
            textbox = tk.Text(self.frame_image, height=1, width=4)
            textbox.insert(tk.END, elem)
            textbox.grid(row=i, column=0)
            self.flagged.append(tk.IntVar(value=0))
            checkbox = tk.Checkbutton(self.frame_image, variable=self.flagged[i], bg='blue')
            checkbox.grid(row=i, column=1)
            for j, elem1 in enumerate(self.images[i]):
                im = Image.open(os.path.join(os.path.dirname(__file__), image_path, elem, elem1))
                width, height = im.size
                cropsize = min(width, height)
                left = (width - cropsize) / 2
                top = (height - cropsize) / 2
                right = (width + cropsize) / 2
                bottom = (height + cropsize) / 2
                im = im.crop((left, top, right, bottom))
                resized = im.resize((80, 80), Image.NEAREST)

                tkimage = ImageTk.PhotoImage(resized)
                im.close()
                myvar = tk.Label(self.frame_image, image=tkimage)
                myvar.image = tkimage
                myvar.grid(row=i, column=j+2)
                myvar.bind("<Button-1>", lambda e, i=i, j=j: self.onClick(i, j))
                if platform.system() == "Darwin":
                    right_button = "<Button-2>"
                    wheel = "<Button-3>"
                else:
                    right_button = "<Button-3>"
                    wheel = "<Button-2>"
                myvar.bind(right_button, lambda e, i=i, j=j: self.onRightClick(i, j))
                myvar.bind(wheel, lambda e, i=i, j=j: self.on_centralClick(i, j))
            print("Scanning folder", elem)

        self.canvas.create_window((0, 0), window=self.frame_image, anchor=tk.NW)
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)
        self.canvas.grid_rowconfigure(0, weight=1)
        self.canvas.grid_columnconfigure(0, weight=1)
        self.frame_image.update_idletasks()
        bbox = self.canvas.bbox(tk.ALL)
        self.canvas.configure(scrollregion=bbox)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheelWin)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview('scroll', -1, 'units')
        elif event.num == 5:
            self.canvas.yview('scroll', 1, 'units')

    def _on_mousewheelWin(self, event):
        if platform.system() == "Darwin":
            self.canvas.yview_scroll(-1 * event.delta, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def onClick(self, i, j):
        selected_img = Image.open(os.path.join(self.image_path, self.image_folders[i], self.images[i][j]))
        width, height = selected_img.size
        cropsize = 20
        left = (width - cropsize) / 2
        top = (height - cropsize) / 2
        right = (width + cropsize) / 2
        bottom = (height + cropsize) / 2
        selected_img = selected_img.crop((left, top, right, bottom))
        resized = selected_img.resize((80, 80), Image.NEAREST)
        new_img = resized
        selected_img.close()
        if self.selected[i][j] == 1:
            enhancer = ImageEnhance.Brightness(resized)
            new_img = enhancer.enhance(0.5)
            self.selected[i][j] = 0
        else:
            self.selected[i][j] = 1
        tkimage = ImageTk.PhotoImage(new_img)
        self.frame_image.grid_slaves(i, j+2)[0].configure(image=tkimage)
        self.frame_image.grid_slaves(i, j+2)[0].image = tkimage

    def onRightClick(self, i, j):
        if all(v == 0 for v in self.selected[i]):
            self.selected[i] = [1 for _ in self.selected[i]]
        else:
            self.selected[i] = [0 for _ in self.selected[i]]

        for x in range(len(self.selected[i])):
            selected_img = Image.open(os.path.join(os.path.dirname(__file__), self.image_path,
                                                   self.image_folders[i], self.images[i][x]))
            width, height = selected_img.size
            cropsize = min(width, height)
            left = (width - cropsize) / 2
            top = (height - cropsize) / 2
            right = (width + cropsize) / 2
            bottom = (height + cropsize) / 2
            selected_img = selected_img.crop((left, top, right, bottom))
            resized = selected_img.resize((80, 80), Image.NEAREST)
            new_img = resized
            selected_img.close()
            if self.selected[i][0] == 0:
                enhancer = ImageEnhance.Brightness(resized)
                new_img = enhancer.enhance(0.5)
            tkimage = ImageTk.PhotoImage(new_img)
            self.frame_image.grid_slaves(i, x+2)[0].configure(image=tkimage)
            self.frame_image.grid_slaves(i, x+2)[0].image = tkimage

    def onButtonClick(self):
        if not os.path.exists(self.result):
            with open(self.result, 'w') as f1:
                f1.write(self.image_folders[-1] + '\n')
                f1.flush()
        else:
            lines = []
            with open(self.result, 'r') as f2:
                lines = f2.readlines()
            lines[0] = self.image_folders[-1] + '\n'
            with open(self.result, 'w') as f3:
                f3.writelines(lines)
                f3.flush()

        with open(self.result, 'a') as file_handler:
            for i, elem in enumerate(self.selected):
                if not all(v == 0 for v in self.selected[i]):
                    mystring = self.image_folders[i] + ';' + \
                               self.frame_image.grid_slaves(i, 0)[0].get("1.0", tk.END).strip() + ';' \
                               + str(self.flagged[i].get()) + ';'

                    for j, elem1 in enumerate(self.selected[i]):
                        if self.selected[i][j] == 1:
                            mystring += self.images[i][j] + ';'
                    mystring = mystring[:-1]
                    mystring += '\n'
                    file_handler.write(mystring)
                    file_handler.flush()
        sys.exit(0)

    def on_centralClick(self, i, j):
        selected_img = Image.open(os.path.join(os.path.dirname(__file__), self.image_path, self.image_folders[i],
                                               self.images[i][j]))
        selected_img.show(title=self.image_folders[i] + '  ' + self.images[i][j])
