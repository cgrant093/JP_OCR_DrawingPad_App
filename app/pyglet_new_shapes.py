
from pyglet.gl import GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA
from pyglet.graphics import Batch, Group
from pyglet.graphics.shader import ShaderProgram
from pyglet.shapes import _get_segment, ShapeBase


class Stroke(ShapeBase):

    def __init__(
            self,
            *coordinates: tuple[float, float],
            thicknesses: list[float],
            color: tuple[int, int, int, int] = (255, 255, 255, 255),
            blend_src: int = GL_SRC_ALPHA,
            blend_dest: int = GL_ONE_MINUS_SRC_ALPHA,
            batch: Batch | None = None,
            group: Group | None = None,
            program: ShaderProgram | None = None,
    ) -> None:
        """Create multiple connected lines from a series of coordinates.

        The shape's anchor point defaults to the first vertex point.

        Similiar to MultiLine, but the line segments can have variable thicknesses
        which is useful for drawing calligraphy characters

        Args:
            coordinates:
                The coordinates for each point in the shape. Each must
                unpack like a tuple consisting of an X and Y float-like
                value.
            thicknesses:
                The list of vertex thicknesses used to calculate 
                the different thicknesses used for the line segments.
                Should have the same number of elements as coordinates
            color:
                The RGB or RGBA color of the shape, specified as a
                tuple of 3 or 4 ints in the range of 0-255. RGB colors
                will be treated as having an opacity of 255.
            blend_src:
                OpenGL blend source mode; for example, ``GL_SRC_ALPHA``.
            blend_dest:
                OpenGL blend destination mode; for example, ``GL_ONE_MINUS_SRC_ALPHA``.
            batch:
                Optional batch to add the shape to.
            group:
                Optional parent group of the shape.
            program:
                Optional shader program of the shape.
        """
        # len(self._coordinates) = the number of vertices in the shape.
        self._rotation = 0
        self._coordinates = list(coordinates)
        self._thicknesses = list(thicknesses)
        self._x, self._y = self._coordinates[0]

        r, g, b, *a = color
        self._rgba = r, g, b, a[0] if a else 255

        super().__init__(
            (len(self._coordinates) - 1) * 6,
            blend_src, blend_dest, batch, group, program,
        )

    def _create_vertex_list(self) -> None:
        self._vertex_list = self._program.vertex_list(
            self._num_verts, self._draw_mode, self._batch, self._group,
            position=('f', self._get_vertices()),
            color=('Bn', self._rgba * self._num_verts),
            translation=('f', (self._x, self._y) * self._num_verts))

    def _get_vertices(self) -> list[float]:
        if not self._visible:
            return (0, 0) * self._num_verts

        trans_x, trans_y = self._coordinates[0]
        trans_x += self._anchor_x
        trans_y += self._anchor_y
        coords: list[list[float]] = [[x - trans_x, y - trans_y] for x, y in self._coordinates]

        # Create a list of triangles from segments between 2 points:
        triangles = []
        prev_miter = None
        prev_scale = None
        for i in range(len(coords) - 1):
            prev_point: list[float] | None = None
            next_point: list[float] | None = None
            if i > 0:
                prev_point = coords[i - 1]

            if i + 2 < len(coords):
                next_point = coords[i + 2]

            prev_miter, prev_scale, *segment = _get_segment(prev_point, coords[i], coords[i + 1], next_point,
                                                            self._thicknesses[i], prev_miter, prev_scale)
            triangles.extend(segment)

        return triangles

    def _update_vertices(self) -> None:
        self._vertex_list.position[:] = self._get_vertices()

    @property
    def thicknesses(self) -> float:
        """Get/set the line thicknesses of the multi-line."""
        return self._thicknesses

    @thicknesses.setter
    def thicknesses(self, thicknesses: list[float]) -> None:
        self._thicknesses = []
        for i in range(len(thicknesses) - 1):
            line_thickness = (thicknesses[i] + thicknesses[i + 1])/2
            self._thicknesses.append(line_thickness)
        # calculated self._thicknesses list should have a length equal to
        #   length(self._coordinates) - 1
        self._update_vertices()

