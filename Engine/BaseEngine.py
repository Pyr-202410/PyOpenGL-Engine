from concurrent.futures import ThreadPoolExecutor

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from pyglm import glm
import math
from PIL import Image
import freetype
from numba import jit



class SDLStyleFontRenderer:
    def __init__(self, engine, font_path, font_size=24, max_atlas_size=1024):
        """
        字体渲染器

        参数:
        engine: 渲染引擎实例
        font_path: 字体文件路径
        font_size: 默认字体大小
        max_atlas_size: 最大图集尺寸
        """
        self.engine = engine
        self.font_path = font_path
        self.default_font_size = font_size
        self.max_atlas_size = max_atlas_size
        
        # 字体缓存
        self.font_cache = {}
        
        # 图集缓存
        self.atlases = {}
        
        # 字符信息缓存
        self.glyph_cache = {}
        
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # 关键：设置1字节对齐
        
        # 加载默认字体
        self.load_font(font_path, font_size)
        self.preload_characters("".join(chr(i) for i in range(32, 127)))
    
    def load_font(self, font_path, font_size=None):
        """加载字体并初始化图集"""
        if font_size is None:
            font_size = self.default_font_size
        
        font_key = (font_path, font_size)
        
        if font_key in self.font_cache:
            return self.font_cache[font_key]
        
        # 创建新的字体缓存
        font_cache = {
            'face': freetype.Face(font_path),
            'size': font_size,
            'atlas_id': None,
            'glyphs': {},
            'max_glyphs': 256,  # 每个图集最大字符数
            'ascender': 0,
            'descender': 0,
            'line_height': 0
        }
        
        # 设置字体大小
        font_cache['face'].set_char_size(font_size * 64)
        
        # 获取字体度量信息
        font_cache['ascender'] = font_cache['face'].size.ascender >> 6
        font_cache['descender'] = font_cache['face'].size.descender >> 6
        font_cache['line_height'] = font_cache['face'].size.height >> 6
        
        # 创建初始图集
        self._create_atlas_for_font(font_cache)
        
        # 缓存字体
        self.font_cache[font_key] = font_cache
        return font_cache
    
    def _create_atlas_for_font(self, font_cache):
        """为字体创建新图集"""
        atlas_id = len(self.atlases)
        atlas = {
            'texture_id': glGenTextures(1),
            'width': self.max_atlas_size,
            'height': self.max_atlas_size,
            'x': 0,
            'y': 0,
            'next_row_height': 0,
            'glyphs': {}
        }
        
        # 初始化纹理
        glBindTexture(GL_TEXTURE_2D, atlas['texture_id'])
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RED,
            atlas['width'] - 1, atlas['height'],
            0, GL_RED, GL_UNSIGNED_BYTE, None
        )
        
        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # 保存图集
        self.atlases[atlas_id] = atlas
        font_cache['atlas_id'] = atlas_id
        return atlas_id
    
    def _add_glyph_to_atlas(self, font_cache, char):
        """添加字符到图集"""
        atlas_id = font_cache['atlas_id']
        atlas = self.atlases[atlas_id]
        
        # 渲染字符
        font_cache['face'].load_char(char, freetype.FT_LOAD_RENDER)
        bitmap = font_cache['face'].glyph.bitmap
        glyph = font_cache['face'].glyph
        
        # 检查空间是否足够 - 修复边界检查
        if atlas['x'] + bitmap.width > atlas['width']:
            # 换行处理
            atlas['x'] = 0
            atlas['y'] += atlas['next_row_height']
            atlas['next_row_height'] = 0
            
            # 检查是否需要新图集 - 修复边界检查
            if atlas['y'] + bitmap.rows >= atlas['height']:  # 改为 >=
                # 创建新图集
                atlas_id = self._create_atlas_for_font(font_cache)
                atlas = self.atlases[atlas_id]
        
        # 确保不会越界 - 添加安全检查
        if (atlas['x'] + bitmap.width > atlas['width'] or
                atlas['y'] + bitmap.rows > atlas['height']):
            # 如果仍然越界，跳过此字符
            print(f"警告: 字符 '{char}' 无法放入图集, 位置: ({atlas['x']}, {atlas['y']}), "
                  f"尺寸: {bitmap.width}x{bitmap.rows}, 图集: {atlas['width']}x{atlas['height']}")
            return None
        
        # 上传字符到位图
        glBindTexture(GL_TEXTURE_2D, atlas['texture_id'])
        
        if bitmap.buffer:
            # 创建临时数组确保数据连续性
            data = np.frombuffer(bytearray(bitmap.buffer), dtype=np.ubyte)
            data = data.reshape(bitmap.rows, bitmap.width)
            glTexSubImage2D(
                GL_TEXTURE_2D, 0,
                atlas['x'], atlas['y'],
                bitmap.width, bitmap.rows,
                GL_RED, GL_UNSIGNED_BYTE,
                data
            )
        
        # 创建字符信息
        glyph_info = {
            'atlas_id': atlas_id,
            'x': atlas['x'],
            'y': atlas['y'],
            'width': bitmap.width,
            'height': bitmap.rows,
            'advance': glyph.advance.x >> 6,
            'bearing_x': glyph.bitmap_left,
            'bearing_y': glyph.bitmap_top
        }
        
        # 更新图集位置
        atlas['x'] += bitmap.width
        if bitmap.rows > atlas['next_row_height']:
            atlas['next_row_height'] = bitmap.rows
        
        # 缓存字符信息
        font_cache['glyphs'][char] = glyph_info
        atlas['glyphs'][char] = glyph_info
        
        glBindTexture(GL_TEXTURE_2D, 0)
        return glyph_info
    
    def get_glyph_info(self, font_path, font_size, char):
        """获取字符信息，如果不存在则创建"""
        font_cache = self.load_font(font_path, font_size)
        
        # 检查是否已缓存
        if char in font_cache['glyphs']:
            return font_cache['glyphs'][char]
        
        # 添加新字符到图集
        return self._add_glyph_to_atlas(font_cache, char)
    
    def render_text(self, text, position, font_path=None, font_size=None,
                    fg_color=(1.0, 1.0, 1.0, 1.0), bg_color=None):
        """
        渲染文本

        参数:
        text: 要渲染的文本
        position: 渲染位置 (x, y)
        font_path: 字体路径 (None使用默认字体)
        font_size: 字体大小 (None使用默认大小)
        fg_color: 前景色 (RGBA)
        bg_color: 背景色 (RGBA) 或 None（透明背景）
        """
        if font_path is None:
            font_path = self.font_path
        if font_size is None:
            font_size = self.default_font_size
        
        font_cache = self.load_font(font_path, font_size)
        
        # 设置使用纹理
        glUseProgram(self.engine.shader)
        glUniform1i(glGetUniformLocation(self.engine.shader, "useTexture"), 1)
        
        # 设置背景色
        if bg_color is None:
            bg_color = (0.0, 0.0, 0.0, 0.0)
        
        bg_loc = glGetUniformLocation(self.engine.shader, "bgColor")
        glUniform4f(bg_loc, *bg_color)
        
        # 获取基线位置
        x, y = position
        y_base = y + font_cache['ascender']
        
        # 创建VAO和VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        
        # 设置顶点属性指针
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # 当前绑定图集ID
        current_atlas_id = -1
        
        # 渲染每个字符
        for char in text:
            glyph_info = self.get_glyph_info(font_path, font_size, char)
            
            # 切换到正确的图集
            if glyph_info['atlas_id'] != current_atlas_id:
                current_atlas_id = glyph_info['atlas_id']
                atlas = self.atlases[current_atlas_id]
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, atlas['texture_id'])
                glUniform1i(glGetUniformLocation(self.engine.shader, "textureSampler"), 0)
            
            # 计算字符位置
            xpos = x + glyph_info['bearing_x']
            # 重要修复：y位置计算
            ypos = y_base - glyph_info['bearing_y']
            
            # 计算纹理坐标 - 修复纹理翻转问题
            atlas = self.atlases[glyph_info['atlas_id']]
            u0 = glyph_info['x'] / atlas['width']
            v0 = glyph_info['y'] / atlas['height']
            u1 = (glyph_info['x'] + glyph_info['width']) / atlas['width']
            v1 = (glyph_info['y'] + glyph_info['height']) / atlas['height']
            
            # 重要修复：顶点位置和纹理坐标
            vertices = np.array([
                # 位置                                                    颜色                 纹理坐标
                # 左下角
                xpos, ypos + glyph_info['height'],                        *fg_color,           u0, v1,
                # 右下角
                xpos + glyph_info['width'], ypos + glyph_info['height'],  *fg_color,           u1, v1,
                # 右上角
                xpos + glyph_info['width'], ypos,                         *fg_color,           u1, v0,
                # 左上角
                xpos, ypos,                                               *fg_color,           u0, v0,
            ], dtype=np.float32)
            
            # 更新VBO数据
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
            
            # 绘制字符 (使用三角形扇)
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
            
            # 更新x位置
            x += glyph_info['advance']
        
        # 清理资源
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
    
    def get_text_size(self, text, font_path=None, font_size=None):
        """获取文本尺寸"""
        if font_path is None:
            font_path = self.font_path
        if font_size is None:
            font_size = self.default_font_size
        
        font_cache = self.load_font(font_path, font_size)
        
        width = 0
        max_ascent = 0
        max_descent = 0
        
        for char in text:
            glyph_info = self.get_glyph_info(font_path, font_size, char)
            width += glyph_info['advance']
            
            # 计算字符的顶部和底部位置
            char_top = glyph_info['bearing_y']
            char_bottom = glyph_info['height'] - glyph_info['bearing_y']
            
            if char_top > max_ascent:
                max_ascent = char_top
            
            if char_bottom > max_descent:
                max_descent = char_bottom
        
        # 使用字体度量信息作为后备
        if max_ascent == 0:
            max_ascent = font_cache['ascender']
        
        if max_descent == 0:
            max_descent = -font_cache['descender']
        
        height = max_ascent + max_descent
        
        return (width, height)
    
    def preload_characters(self, text, font_path=None, font_size=None):
        """预加载文本中的字符"""
        if font_path is None:
            font_path = self.font_path
        if font_size is None:
            font_size = self.default_font_size
        
        font_cache = self.load_font(font_path, font_size)
        
        for char in set(text):
            if char not in font_cache['glyphs']:
                self.get_glyph_info(font_path, font_size, char)
    
    def cleanup(self):
        """清理资源"""
        for atlas_id, atlas in self.atlases.items():
            glDeleteTextures(1, [atlas['texture_id']])
        
        self.atlases = {}
        self.font_cache = {}
        self.glyph_cache = {}


class BaseWindow:
    """窗口管理类，负责窗口创建、事件处理和生命周期管理"""
    
    def __init__(self, width=800, height=600, title="Game Window"):
        # 初始化GLFW
        if not glfw.init():
            raise RuntimeError("GLFW初始化失败")
        
        # 设置OpenGL版本
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        # 创建窗口
        self._width = width
        self._height = height
        self._window = glfw.create_window(width, height, title, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("GLFW Window create Error")
        
        # 设置当前上下文
        glfw.make_context_current(self._window)
        
        # 设置视口
        glViewport(0, 0, width, height)
        
        # 设置窗口大小改变回调
        glfw.set_window_size_callback(self._window, self._on_resize)
        
        # 存储回调函数
        self.resize_callback = None
    
    def _on_resize(self, window, width, height):
        """窗口大小改变回调函数"""
        # 更新窗口尺寸
        self._width = width
        self._height = height
        
        # 更新视口
        glViewport(0, 0, width, height)
        
        # 如果有外部回调，则执行
        if self.resize_callback:
            self.resize_callback(width, height)
    
    def set_resize_callback(self, callback):
        """设置窗口大小改变回调函数"""
        self.resize_callback = callback
    
    def isQuit(self):
        """检查窗口是否应该关闭"""
        return glfw.window_should_close(self._window)
    
    def poll_events(self):
        """处理事件"""
        glfw.poll_events()
    
    def swap_buffers(self):
        """交换缓冲区"""
        glfw.swap_buffers(self._window)
    
    def getWindowSize(self):
        return glfw.get_window_size(self._window)
    
    def getCursorPos(self):
        return glfw.get_cursor_pos(self._window)
    
    def close(self):
        """终止GLFW"""
        glfw.terminate()
    


class Base2DEngine:
    """渲染引擎类，负责所有渲染操作"""
    
    def __init__(self, window):
        # 保存窗口引用
        self.window = window
        
        # 设置OpenGL状态
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.1, 1.0)
        
        # 初始化着色器
        self._init_shaders()
        
        # 设置投影矩阵
        self._update_projection()
        
        # 设置窗口大小改变回调
        window.set_resize_callback(self._on_window_resize)
        
        # 纹理缓存
        self.textures = {}
        
        # 创建SDL风格字体渲染器
        self.font_renderer = SDLStyleFontRenderer(self, "msyh.ttc", font_size=24)
        
    
    def _init_shaders(self):
        """初始化着色器程序"""
        # 顶点着色器
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec2 position;    // 顶点位置
        layout (location = 1) in vec4 color;       // 顶点颜色
        layout (location = 2) in vec2 texCoord;    // 纹理坐标

        uniform mat4 projection;                   // 正交投影矩阵

        out vec4 fragColor;                        // 传递给片段着色器的颜色
        out vec2 fragTexCoord;                     // 传递给片段着色器的纹理坐标

        void main() {
            // 计算最终位置
            gl_Position = projection * vec4(position, 0.0, 1.0);
            fragColor = color;
            fragTexCoord = texCoord;
        }
        """
        
        # 片段着色器
        fragment_shader = """
                #version 330 core
                in vec4 fragColor;
                in vec2 fragTexCoord;

                uniform sampler2D textureSampler;
                uniform int useTexture;
                uniform vec4 bgColor;  // 新增背景色uniform

                out vec4 outColor;

                void main() {
                    if (useTexture == 1) {
                        // 使用纹理：采样纹理并乘以颜色
                        vec4 texColor = texture(textureSampler, fragTexCoord);
                        float alpha = texColor.r;  // 使用红色通道作为alpha

                        // 混合前景色和背景色
                        vec4 fg = vec4(fragColor.rgb, fragColor.a * alpha);
                        vec4 bg = vec4(bgColor.rgb, bgColor.a * (1.0 - alpha));

                        outColor = mix(bg, fg, alpha);
                    } else {
                        // 不使用纹理：直接使用顶点颜色
                        outColor = fragColor;
                    }
                }
                """
        
        # 编译着色器
        vert_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
        frag_shader = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        
        # 链接着色器程序
        self.shader = compileProgram(vert_shader, frag_shader)
    
    def _on_window_resize(self, width, height):
        """窗口大小改变回调"""
        self._update_projection()
    
    def _update_projection(self):
        """更新投影矩阵"""
        width, height = glfw.get_window_size(self.window._window)
        
        # 创建正交投影矩阵（将世界坐标映射到屏幕坐标）
        self.projection = glm.ortho(0, width, height, 0, -1.0, 1.0)
        
        # 使用着色器程序
        glUseProgram(self.shader)
        
        # 获取投影矩阵在着色器中的位置
        proj_loc = glGetUniformLocation(self.shader, "projection")
        
        # 设置投影矩阵uniform变量
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(self.projection))
    
    def begin_frame(self):
        """开始一帧渲染"""
        # 清除颜色缓冲区
        glClear(GL_COLOR_BUFFER_BIT)
        
        # 使用着色器程序
        glUseProgram(self.shader)
    
    def end_frame(self):
        """结束一帧渲染"""
        # 交换缓冲区
        self.window.swap_buffers()
    
    def draw_line(self, start, end, color=(1.0, 1.0, 1.0, 1.0), width=1.0):
        """绘制线条"""
        # 设置不使用纹理
        glUniform1i(glGetUniformLocation(self.shader, "useTexture"), 0)
        
        # 设置线宽
        glLineWidth(width)
        
        # 创建顶点数据（位置、颜色、纹理坐标）
        vertices = np.array([
            start[0], start[1], *color, 0.0, 0.0,  # 起点
            end[0], end[1], *color, 1.0, 1.0  # 终点
        ], dtype=np.float32)
        
        # 创建顶点数组对象(VAO)
        vao = glGenVertexArrays(1)
        # 创建顶点缓冲对象(VBO)
        vbo = glGenBuffers(1)
        
        # 绑定VAO
        glBindVertexArray(vao)
        # 绑定VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        # 将顶点数据复制到缓冲区
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # 设置顶点属性指针
        # 位置属性 (location=0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # 颜色属性 (location=1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        # 纹理坐标属性 (location=2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # 绘制线条
        glDrawArrays(GL_LINES, 0, 2)
        
        # 清理资源
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
    
    def draw_polygon(self, vertices, fill=None, border=None, width=1.0):
        """
        绘制多边形
        - vertices: 顶点列表 [(x1,y1), (x2,y2), ...]
        - fill: 填充颜色 (R,G,B,A) 或 None
        - border: 边框颜色 (R,G,B,A) 或 None
        - width: 边框宽度
        """
        # 绘制填充部分
        if fill is not None:
            self._draw_filled_polygon(vertices, fill)
        
        # 绘制边框
        if border is not None and len(vertices) >= 2:
            for i in range(len(vertices)):
                start = vertices[i]
                end = vertices[(i + 1) % len(vertices)]
                if width:
                    self.draw_line(start, end, border, width)
    
    def _draw_filled_polygon(self, vertices, color):
        """绘制填充多边形"""
        # 设置不使用纹理
        glUniform1i(glGetUniformLocation(self.shader, "useTexture"), 0)
        
        # 创建顶点数据
        vert_data = []
        for v in vertices:
            vert_data.extend([v[0], v[1], *color, 0.0, 0.0])
        vert_data = np.array(vert_data, dtype=np.float32)
        
        # 创建索引数据（三角形扇）
        indices = []
        for i in range(1, len(vertices) - 1):
            indices.extend([0, i, i + 1])
        indices = np.array(indices, dtype=np.uint32)
        
        # 创建顶点数组对象(VAO)
        vao = glGenVertexArrays(1)
        # 创建顶点缓冲对象(VBO)
        vbo = glGenBuffers(1)
        # 创建元素缓冲对象(EBO)
        ebo = glGenBuffers(1)
        
        # 绑定VAO
        glBindVertexArray(vao)
        
        # 绑定VBO并设置顶点数据
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vert_data.nbytes, vert_data, GL_STATIC_DRAW)
        
        # 绑定EBO并设置索引数据
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # 设置顶点属性指针
        # 位置属性 (location=0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # 颜色属性 (location=1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        # 纹理坐标属性 (location=2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # 绘制多边形
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        
        # 清理资源
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glDeleteBuffers(1, [ebo])
    
    def draw_rectangle(self, pos, size, fill=None, border=None, width=1.0):
        """
        绘制矩形
        - pos: 左上角位置 (x,y)
        - size: 尺寸 (width, height)
        - fill: 填充颜色 (R,G,B,A) 或 None
        - border: 边框颜色 (R,G,B,A) 或 None
        - width: 边框宽度
        """
        # 计算矩形顶点
        x, y = pos
        w, h = size
        vertices = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        
        # 使用多边形绘制矩形
        self.draw_polygon(vertices, fill, border, width)
    
    def draw_arc(self, center, radius, start_angle, end_angle, fill=None, border=None, width=1.0):
        """绘制弧形"""
        cx, cy = center
        segments = int(max(16, abs(end_angle - start_angle) / 5))
        points = []
        
        # 计算弧线上的点
        for i in range(segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / segments
            rad = math.radians(angle)
            x = cx + radius * math.cos(rad)
            y = cy + radius * math.sin(rad)
            points.append((x, y))
        
        # 绘制填充部分
        if fill is not None:
            filled_points = [center] + points
            self._draw_filled_polygon(filled_points, fill)
        
        # 绘制边框
        if border is not None:
            for i in range(len(points) - 1):
                self.draw_line(points[i], points[i + 1], border, width)
    
    def create_texture(self, image_path):
        """从图像文件创建纹理"""
        # 打开图像文件
        img = Image.open(image_path)
        # 转换为RGBA格式
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 将图像数据转换为NumPy数组
        img_data = np.array(img, dtype=np.uint8)
        
        # 生成纹理ID
        texture_id = glGenTextures(1)
        # 绑定纹理
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # 设置纹理数据
        glTexImage2D(
            GL_TEXTURE_2D,  # 目标纹理（2D纹理）
            0,  # Mipmap级别（0表示基本级别）
            GL_RGBA,  # 纹理内部格式
            img.width,  # 纹理宽度
            img.height,  # 纹理高度
            0,  # 边框（必须为0）
            GL_RGBA,  # 像素数据格式
            GL_UNSIGNED_BYTE,  # 像素数据类型
            img_data  # 像素数据
        )
        
        # 设置纹理参数
        # 纹理环绕方式（S方向）
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        # 纹理环绕方式（T方向）
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # 纹理缩小过滤方式
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # 纹理放大过滤方式
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # 解绑纹理
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # 缓存纹理ID
        self.textures[image_path] = texture_id
        return texture_id
    
    def draw_texture(self, pos, size, texture_id, color=(1.0, 1.0, 1.0, 1.0)):
        """绘制纹理"""
        # 设置使用纹理
        glUniform1i(glGetUniformLocation(self.shader, "useTexture"), 1)
        
        x, y = pos
        w, h = size
        
        # 创建顶点数据
        vertices = np.array([
            x, y, *color, 0.0, 0.0,  # 左上
            x + w, y, *color, 1.0, 0.0,  # 右上
            x + w, y + h, *color, 1.0, 1.0,  # 右下
            x, y + h, *color, 0.0, 1.0  # 左下
        ], dtype=np.float32)
        
        # 索引数据（两个三角形组成矩形）
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # 创建顶点数组对象(VAO)
        vao = glGenVertexArrays(1)
        # 创建顶点缓冲对象(VBO)
        vbo = glGenBuffers(1)
        # 创建元素缓冲对象(EBO)
        ebo = glGenBuffers(1)
        
        # 绑定VAO
        glBindVertexArray(vao)
        
        # 绑定VBO并设置顶点数据
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # 绑定EBO并设置索引数据
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # 设置顶点属性指针
        # 位置属性 (location=0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # 颜色属性 (location=1)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        # 纹理坐标属性 (location=2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        # 绑定纹理
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        # 设置纹理采样器uniform
        glUniform1i(glGetUniformLocation(self.shader, "textureSampler"), 0)
        
        # 绘制图像
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # 清理资源
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])
        glDeleteBuffers(1, [ebo])
    
    def preload_font_characters(self, text, font_path, font_size):
        """
        预加载字体字符集
        """
        self.font_renderer.preload_characters(text,font_path, font_size)
    
    def get_text_size(self, text, font_path, font_size=24):
        """
        获取文本尺寸（基于SDL）
        """
        return self.font_renderer.get_text_size(text,font_path, font_size)
    
    def draw_text(self, text, position, font_path, font_size=24,
                  fg_color=(1.0, 1.0, 1.0, 1.0), bg_color=None):
        """
        绘制文字（基于SDL）
        """
        if bg_color:
            self.draw_rectangle(position,self.get_text_size(text,font_path, font_size),bg_color,None,0)
        self.font_renderer.render_text(text,position, font_path, font_size, fg_color, bg_color)
    
    def cleanup(self):
        """清理资源"""
        # 删除着色器程序
        glDeleteProgram(self.shader)
        
        # 删除所有纹理
        for texture_id in self.textures.values():
            glDeleteTextures(1, [texture_id])
            
        # 删除所有字体纹理
        self.font_renderer.cleanup()


class ResLoadThread(ThreadPoolExecutor):
    class Mode:
        ...
    def __init__(self):
        super().__init__(max_workers = 8,thread_name_prefix = "RedLoad")
        
    def submit(self,mode,**kwargs):
        ...