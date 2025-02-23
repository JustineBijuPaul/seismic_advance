import os
from PIL import Image, ImageDraw

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Create a new image with a transparent background
size = (256, 256)
icon = Image.new('RGBA', size, (0, 0, 0, 0))
draw = ImageDraw.Draw(icon)

# Draw a simple seismic wave icon
center_x, center_y = size[0] // 2, size[1] // 2
wave_color = (255, 68, 68, 255)  # Red color
line_width = 8

# Draw seismic wave lines
points = [
    (50, center_y),
    (100, center_y - 50),
    (150, center_y + 50),
    (206, center_y)
]

draw.line(points, fill=wave_color, width=line_width)

# Save the icon
icon.save('static/icon.png', 'PNG')
print("Icon created successfully at static/icon.png")
