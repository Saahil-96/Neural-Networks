from PIL import Image

for num in range(12629):
    if num<10:
        im=Image.open("Images/0000"+str(num)+".ppm")
        im.save("New Dataset/0000"+str(num)+".jpg")
    elif num<100:
        im=Image.open("Images/000"+str(num)+".ppm")
        im.save("New Dataset/000"+str(num)+".jpg")
    elif num<1000:
        im=Image.open("Images/00"+str(num)+".ppm")
        im.save("New Dataset/00"+str(num)+".jpg")
    elif num<10000:
        im=Image.open("Images/0"+str(num)+".ppm")
        im.save("New Dataset/0"+str(num)+".jpg")
    else:
        im=Image.open("Images/"+str(num)+".ppm")
        im.save("New Dataset/"+str(num)+".jpg")


