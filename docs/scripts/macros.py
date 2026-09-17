# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

import json


def define_env(env):
    @env.macro
    def mol3d(href, type="xyz", height="25em", backgroundalpha=0, style=None, ui=True):
        """Render a 3Dmol.js viewer div for a structure file."""
        if style is None:
            style = {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}}

        return f"""<div style="height: {height}; position: relative;"
     class="viewer_3Dmoljs"
     data-href="{href}"
     data-type="{type}"
     data-backgroundalpha="{backgroundalpha}"
     data-style='{json.dumps(style)}'
     data-ui="{str(ui).lower()}">
</div>"""
