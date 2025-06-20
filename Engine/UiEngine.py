# 在BaseEngine.py基础上添加以下内容
import queue
import time

from OpenGL.GL import *
from glfw import MOUSE_BUTTON_LEFT

from BaseEngine import BaseWindow,Base2DEngine
import glfw


baseFont = "msyh.ttc"

def checkPrimaryWindow():
    if primaryWindow is None:
        class NoPrimaryWindowError(Exception): pass
        raise NoPrimaryWindowError

class Window(BaseWindow):
    def __init__(self,title,width = 800,height = 600,bg = (1,1,1,1),icon = None):
        global primaryWindow
        super().__init__(width, height, title)
        self._TYPE = "Window"
        self.engine = Base2DEngine(self)
        primaryWindow = self
        self.bg = bg
        
        if icon:
            glfw.set_window_icon(self._window,1,"../icon.png")
        
        self.widgets = []
        '''[
              {
                "obj": "<class>",
                #可选: [
                    "children": [
                        {...}
                      ]
                    }
                ]#
              }
            ]
        '''
        self._press = []
        self._release = []
        self._button = []
        self._input = []
        
        self._key = []

        self.events = queue.Queue()
        glfw.set_window_pos_callback(self._window,           lambda *args:self._getEvent("window_pos",*args[1:]))
        glfw.set_window_close_callback(self._window,         lambda *args:self._getEvent("window_close",*args[1:]))
        glfw.set_window_refresh_callback(self._window,       lambda *args:self._getEvent("window_refresh",*args[1:]))
        glfw.set_window_focus_callback(self._window,         lambda *args:self._getEvent("window_focus",*args[1:]))
        glfw.set_window_iconify_callback(self._window,       lambda *args:self._getEvent("window_iconify",*args[1:]))
        glfw.set_window_maximize_callback(self._window,      lambda *args:self._getEvent("window_maximize",*args[1:]))
        glfw.set_framebuffer_size_callback(self._window,     lambda *args:self._getEvent("framebuffer_size",*args[1:]))
        glfw.set_window_content_scale_callback(self._window, lambda *args:self._getEvent("window_content_scale",*args[1:]))
        glfw.set_key_callback(self._window,                  lambda *args:self._getEvent("key",*args[1:]))
        glfw.set_char_callback(self._window,                 lambda *args:self._getEvent("char",*args[1:]))
        glfw.set_mouse_button_callback(self._window,         lambda *args:self._getEvent("mouse_button",*args[1:]))
        glfw.set_cursor_pos_callback(self._window,           lambda *args:self._getEvent("cursor_pos",*args[1:]))
        glfw.set_cursor_enter_callback(self._window,         lambda *args:self._getEvent("cursor_enter",*args[1:]))
        glfw.set_scroll_callback(self._window,               lambda *args:self._getEvent("scroll",*args[1:]))
        glfw.set_drop_callback(self._window,                 lambda *args:self._getEvent("drop",*args[1:]))
        glfw.set_char_mods_callback(self._window,            lambda *args:self._getEvent("char_mods",*args[1:]))
        
    def update(self,widgets = None):
        self.engine.draw_rectangle((0,0),self.getWindowSize(),self.bg,None,0)
        if widgets is None:
            widgets = self.widgets
        for w in widgets:
            obj = w["obj"]
            winSize = self.getWindowSize()
            if obj.x > winSize[0] or obj.y > winSize[1]:
                continue
            if obj._TYPE == "Button":
                obj.update(obj._state)
            else:
                obj.update()
            if "children" in w:
                self.update(w["children"])
            
    
    def mainloop(self):
        last = time.time()
        step = 0
        glfw.swap_interval(0)
        fps = 0
        while not self.isQuit():
            # 处理事件
            self.poll_events()
            
            while not self.events.empty():
                event = self.events.get()
                print(event)
                if event[0] == "window_refresh":
                    refresh = True
                elif event[0] == "mouse_button" and event[1] == glfw.MOUSE_BUTTON_LEFT:
                    cur = self.getCursorPos()
                    if event[2] == glfw.PRESS:
                        data = self._press
                        for b in self._button:
                            pos = b[0]
                            size = b[1]
                            btn = b[2]
                            if pos[0] <= cur[0] <= pos[0] + size[0] and pos[1] <= cur[1] <= pos[1] + size[1]:
                                btn._state = 1
                                break
                        for i in self._input:
                            pos = i[0]
                            size = i[1]
                            inputs = i[2]
                            if pos[0] <= cur[0] <= pos[0] + size[0] and pos[1] <= cur[1] <= pos[1] + size[1]:
                                inputs._on_focus = True
                                inputs._click_seek(cur[0])
                                break
                            else:
                                inputs._on_focus = False
                    elif event[2] == glfw.RELEASE:
                        data = self._release
                        for b in self._button:
                            if b[2]._state:
                                b[2]._state = 0
                                break
                    else:
                        continue
                    for b in data:
                        pos = b[0]
                        size = b[1]
                        callback = b[2]
                        if pos[0] <= cur[0] <= pos[0] + size[0] and pos[1] <= cur[1] <= pos[1] + size[1]:
                            callback()
                            break
                elif event[0] == "char":
                    for i in self._input:
                        if i[2]._on_focus:
                            i[2]._add_char(chr(event[1]))
                            break
                elif event[0] == "key":
                    if event[3] == 1:
                        self._key.append(glfw.get_key_name(event[1],event[2]))
                    for i in self._input:
                        if i[2]._on_focus and event[3] in (glfw.PRESS,glfw.REPEAT):
                            if event[1] == glfw.KEY_BACKSPACE:
                                i[2]._del_char("l")
                            elif event[1] == glfw.KEY_DELETE:
                                i[2]._del_char("r")
                            elif event[1] == glfw.KEY_LEFT:
                                i[2]._direct_seek("l")
                            elif event[1] == glfw.KEY_RIGHT:
                                i[2]._direct_seek("r")
                            elif event[1] == glfw.KEY_UP:
                                i[2]._direct_seek("u")
                            elif event[1] == glfw.KEY_DOWN:
                                i[2]._direct_seek("d")
                            elif event[1] == glfw.KEY_LEFT_CONTROL | glfw.KEY_V:
                                print("ctrl+v3")
                                glfw.MOD_CONTROL
            # 开始帧渲染
            self.engine.begin_frame()
            
            # 绘制各种图形
            self.update()
            
            if time.time() - step >= 1:
                d = time.time() - last
                if d > 0:
                    fps = round(1 / d, 2)
                    step = time.time()
            self.engine.draw_text("FPS: " + str(fps), (0, 0), "msyh.ttc", 20, (0, 0, 0, 1), (1, 1, 1, 1))
            # 结束帧渲染
            self.engine.end_frame()
            last = time.time()
        
            
        # 清理资源
        self.engine.cleanup()
        self.close()
    
    def _getEvent(self,*args):
        self.events.put(args)
    
    def _on_resize(self, *args):
        super()._on_resize(*args)
        self._getEvent("resize",*args)
    
    def _resignedPress(self,pos,size,callback):
        self._press.append((pos,size,callback))
    
    def _resignedRelease(self,pos,size,callback):
        self._release.append((pos,size,callback))
    
    def _resignedButton(self,pos,size,button):
        self._button.append((pos,size,button))
    
    def _resignedInput(self,pos,size,inputs):
        self._input.append((pos,size,inputs))
    

primaryWindow: Window = ...


class Widget:
    def __init__(self,parent):
        checkPrimaryWindow()
        self._TYPE = "Widget"
        self.parent = parent
        self.engine: Base2DEngine = parent.engine
        if parent == primaryWindow:
            primaryWindow.widgets.append({"obj":self})
            self.path = []
        else:
            t = primaryWindow.widgets
            for i in self.parent.path:
                t = t[i]
            t[self.parent._TYPE].append({"obj":self})
    
    def update(self):
        pass # 实际组件内完成
    

class Label(Widget):
    def __init__(self,parent,x,y,text = "Label",fontsize = 20,font = baseFont,bg = (0,0,0,0),fg = (0,0,0,1)):
        super().__init__(parent)
        self._TYPE = "Label"
        self.text = text
        self.fontsize = fontsize
        self.font = font
        self.x,self.y = x,y
        self.bg = bg
        self.fg = fg
    
    def update(self):
        self.engine.draw_text(self.text,(self.x,self.y),self.font,self.fontsize,self.fg,self.bg)

class Button(Widget):
    def __init__(self,parent,x,y,text = "Button",fontsize = 20,font = baseFont,bg = (0,0,0,0.5),fg = (1,1,1,1),press_bg = (0,0,0,0.6),bd = (0.2,0.2,0.2,0.8),bd_size = 3,press = None,release = None,side = 5):
        super().__init__(parent)
        self._TYPE = "Button"
        self.text = text
        self.fontsize = fontsize
        self.font = font
        self.x,self.y = x,y
        self.bg = bg
        self.fg = fg
        self.press_bg = press_bg
        self.bd = bd
        self.bd_size = bd_size
        self.side = side
        self.press_callback = press
        self.release_callback = release
        self._state = 0
        
        self._size = self.engine.get_text_size(self.text,self.font,self.fontsize)
        self._size = (self._size[0] + self.side * 2,self._size[1] + self.side * 2)
        if self.press_callback:
            primaryWindow._resignedPress((x,y),self._size,self.press_callback)
        if self.release_callback:
            primaryWindow._resignedRelease((x,y),self._size,self.release_callback)
        primaryWindow._resignedButton((x,y),self._size,self)
    
    def update(self,state = 0):
        if state == 1: # 按下
            self.engine.draw_rectangle((self.x, self.y),
                                       self._size, self.press_bg,
                                       self.bd, self.bd_size)
            self.engine.draw_text(self.text, (self.x + self.side, self.y + self.side), self.font, self.fontsize,
                                  self.fg, None)
        else: # 默认
            self.engine.draw_rectangle((self.x, self.y),
                                       self._size, self.bg,
                                       self.bd, self.bd_size)
            self.engine.draw_text(self.text, (self.x + self.side, self.y + self.side), self.font, self.fontsize,
                                  self.fg, None)


class Entry(Widget):
    def __init__(self,parent,x,y,fontsize = 20,font = baseFont,bg = (1,1,1,1),fg = (0,0,0,1),bd = (0,0,0,1),bd_size = 1,focus_bd = (0,0,0,0.7),width = 100):
        super().__init__(parent)
        self._TYPE = "Entry"
        self.fontsize = fontsize
        self.font = font
        self.x, self.y = x, y
        self.bg = bg
        self.fg = fg
        self.bd = bd
        self.bd_size = bd_size
        self.focus_bd = focus_bd
        self.width = width
        
        self.text = [
                     ] # 单个字符: {"char":char,"width":self.engine.get_text_size(char,self.font,self.fontsize)[0]}
        self._cur = 0
        self._show = [0,0]
        self._on_focus = False
        self._draw_cur_info = {"lastUpd":time.time(),"state":False}
        
        self._size = (self.width,self.engine.get_text_size("|",self.font,self.fontsize)[1] + 5)
        
        primaryWindow._resignedInput((self.x,self.y),self._size,self)
        
    
    def update(self):
        bd = self.focus_bd if self._on_focus else self.bd
        self.engine.draw_rectangle((self.x,self.y),self._size,self.bg,bd,1)
        if self._on_focus:
            now = time.time()
            if now - self._draw_cur_info["lastUpd"] >= 0.5:
                self._draw_cur_info["state"] = not self._draw_cur_info["state"]
                self._draw_cur_info["lastUpd"] = now
            t = self.text[self._show[0]:self._cur]
            x = self.x + sum(map(lambda x:x["width"],t))
            if self._draw_cur_info["state"]:
                self.engine.draw_line((x,self.y + 2),(x,self.y + self._size[1] - 2),self.fg,1)
        t = self.text[self._show[0]:self._show[1] + 1]
        self.engine.draw_text("".join(map(lambda x:x["char"],t)),(self.x,self.y),self.font,self.fontsize,self.fg,None)
    
    #print(self._cur,"".join(map(lambda x:x["char"],self.text)))
    def _add_char(self,char):
        char_width = self.engine.get_text_size(char,self.font,self.fontsize)[0]
        self.text.insert(self._cur,{"char":char,"width":char_width})
        self._cur += 1
        width = sum(map(lambda x: x["width"], (t := self.text[self._show[0]:self._show[1] + 1])))
        if width >= self.width:
            self._show[0] += 1
            self._show[1] += 1
        else:
            self._show[1] += 1
        self._out_check("add")
        self._draw_cur_info["state"] = True
    
    def _del_char(self,side = "l"):
        if self._cur > 0:
            if side == "l":
                if 0 <= self._cur - 1 < len(self.text): # 再次确认
                    self.text.pop(self._cur - 1)
                self._cur -= 1
                if self._show[0] >= 1:
                    self._show = [self._show[0] - 1,self._show[1] - 1]
            else:
                if 0 <= self._cur < len(self.text):  # 再次确认
                    self.text.pop(self._cur)
            self._out_check("add")
        self._draw_cur_info["state"] = True
    
    def _click_seek(self,x):
        # 基于距离的光标点击定位算法（未优化）
        x = x - self.x
        width = 0
        ws = []
        for c in self.text[self._show[0]:self._show[1] + 1]:
            ws.append(abs(width - x))
            width += c["width"]
        for i,w in enumerate(ws):
            if i + 1 < len(ws):
                nw = ws[i + 1]
                if nw > w:
                    self._cur = self._show[0] + i
                    break
            else:
                if w < self.text[-1]["width"]:
                    self._cur = self._show[0] + i
                else:
                    self._cur = self._show[1]
        self._draw_cur_info["state"] = True
    
    def _direct_seek(self,d):
        if d == "l":
            if self._cur > 0:
                self._cur -= 1
        elif d == "r":
            if self._cur < len(self.text):
                self._cur += 1
        
    
    def _out_check(self,mode = "add"):
        while sum(map(lambda x: x["width"], (t := self.text[self._show[0]:self._show[1] + 1]))) > self.width:
            if not self._show[0] <= self._cur <= self._show[1]:
                break
            if mode == "add":
                self._show[0] += 1
            else:
                self._show[1] -= 1

if __name__ == '__main__':
    
    win = Window("hello",bg = (0,0,0,0))
    '''
    for x in range(0,1900,200):
        for y in range(0,1050,50):
            Button(win,x,y,press = lambda:print("Click!"),release = lambda:print("up"))
    '''
    Entry(win,10,100,width = 500)
    win.mainloop()