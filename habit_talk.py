from tkinter import*
from tkinter import messagebox

window = Tk()
window.title("Habittalk")
window.geometry("500x300+500+250")

label_input = Label(window, text = "Input", font = ("궁서체", 15))
input_data = Entry(window, width = 200)
label_input.pack()
input_data.pack()
input_data.focus_set()


def output_data():
    print (input_data.get()) # This is the text you may want to use later

    label_output = Label(window, text = "Output", font =  ("궁서체", 15))
    label1 = Label(window, text = "Input : " + input_data.get(), font = ("굴림체", 12), fg = "white", bg = "Gray", width = 150)
    label2 = Label(window, text = "Output : " + input_data.get() + "'s output is a", font = ("굴림체", 12), bg = "white", width = 150)
    label_output.pack()
    label1.pack()
    label2.pack()

ok_btn = Button(window, text = "OK", width = 10, command = output_data)
exit_Button = Button(window, text = 'Exit', width = 10, command = window.destroy)

ok_btn.pack()
exit_Button.pack()

window.mainloop()
