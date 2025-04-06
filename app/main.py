

from util import MyApp, MyTablet, CalligraphyCharacter

import math

# importing pyglet module 
import pyglet
 
# importing shapes from the pyglet
from pyglet import shapes


# # (japcharapp) F:\projects\jp_ocr_drawingpad_app\JP_OCR_DrawingPad_App\app


 

def event_loop():
    
    # creating batch objects
    bg_batch = pyglet.graphics.Batch()
    char_batch = pyglet.graphics.Batch()

    # creating a window 
    app = MyApp(bg_batch)

    tablet = MyTablet(app)

    callig_char = CalligraphyCharacter(app.draw_area)

    
    # tablet.set_window_attrs(app.x_factor, app.sx, app.sy)
    
    black = (0, 0, 0)

    global cheese
    cheese = 1

    # @tablet.contact.event
    # def on_press():
    #     if tablet.pen_btn.value:
    #         pyglet.event.EVENT_HANDLED
    #     # callig_char.new_stroke()
    #     # data = tablet.get_data()
    #     # if data['pressure'] <= 0:
    #     #     pyglet.event.EVENT_HANDLED
    #     # if data['coordinate'] in app.draw_area:
    #     #     callig_char.new_stroke()
        
    # @tablet.contact.event
    # def on_release():
    #     if tablet.pen_btn.value:
    #         pyglet.event.EVENT_HANDLED
    #     callig_char.finish_stroke(app.draw_area)

    @tablet.P.event
    def on_change(value):
        global cheese

        if not tablet.contact.value:
            pyglet.event.EVENT_HANDLED

        data = tablet.get_data()
        if data['coordinate'] not in app.draw_area:
            pyglet.event.EVENT_HANDLED

        if tablet.pen_btn.value:
            pyglet.event.EVENT_HANDLED

        if value == 0:
            callig_char.finish_stroke()
            callig_char.new_stroke()

        if value > 0:
            callig_char.append_stroke_data(data)
            callig_char.draw_stroke(char_batch, color=black)

        
        # in_contact = (value > 0) and tablet.contact.value
        # if in_contact and data['coordinate'] in app.draw_area:
        #     if not tablet.pen_btn.value:
        #         callig_char.append_stroke_data(data)
        #         callig_char.draw_stroke(app.draw_area, char_batch, color=black)

    @tablet.pen_btn.event
    def on_press():
        callig_char.clear_all()
            


    # @app.window.event
    # def on_mouse_press(x, y, button, modifiers):
    #     if button == pyglet.window.mouse.LEFT:
    #         global is_drawing
    #         is_drawing = True
            
    # @app.window.event
    # def on_mouse_release(x, y, button, modifiers):
    #     if button == pyglet.window.mouse.LEFT:
    #         global is_drawing
    #         is_drawing = False
    #         points.append(None)

    # @app.window.event
    # def on_mouse_drag(x, y, dx, dy, button, modifiers):
    #     if is_drawing and is_pen_drawing:
    #         pos = (x, y)
    #         if pos not in points:
    #             points.append(pos)
    

    # @app.window.event
    # def on_resize(width, height):
    #     app.calc_draw_area(bg_batch)
    #     tablet.set_window_attrs(app.x_factor, app.sx, app.sy)
    
    # window draw event
    @app.window.event
    def on_draw():
        app.window.clear()

        # draw the background batch
        bg_batch.draw()
        
        # draw the character batch
        char_batch.draw()
    
    # run the pyglet application
    app.run()


if __name__ == '__main__':
    event_loop()