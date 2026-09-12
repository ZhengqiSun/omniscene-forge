from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path


def probe(path):
    out=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=width,height,r_frame_rate,nb_read_frames","-of","json",path],text=True)
    return json.loads(out)["streams"][0]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--mode",choices=("native","candidate101_to_81x16"),required=True); a=p.parse_args()
    out=Path(a.output)
    if out.exists(): raise FileExistsError(out)
    if a.mode=="native": args=["-vf","scale=832:480,fps=20","-frames:v","81"]
    else: args=["-vf","scale=832:480,fps=16","-frames:v","81"]
    subprocess.run(["ffmpeg","-v","error","-i",a.input,*args,"-c:v","libx264","-pix_fmt","yuv420p",str(out)],check=True)
    print(json.dumps(probe(str(out)),indent=2))


if __name__ == "__main__": main()
