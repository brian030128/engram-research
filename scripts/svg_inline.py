import re,sys
p=sys.argv[1]; s=open(p,encoding='utf8').read()
A={'box':'fill="#ffffff" stroke="#3d4654" stroke-width="1.4" rx="7"','det':'fill="#e9f1fb" stroke="#2f6db3" stroke-width="1.4" rx="7"','mem':'fill="#fdf1dd" stroke="#b5791a" stroke-width="1.6" rx="7"','gate':'fill="#fbe8ea" stroke="#b53a4b" stroke-width="1.6" rx="7"','opt':'fill="#f1f2f4" stroke="#7a828e" stroke-width="1.3" stroke-dasharray="5 4" rx="7"','dyn':'fill="#e8f5ec" stroke="#2f8a55" stroke-width="1.4" rx="7"','t':'font-size="14.5" font-weight="700" fill="#1c2430" text-anchor="middle"','s':'font-size="12" fill="#3d4654" text-anchor="middle"','n':'font-size="12" fill="#59626f"','h':'font-size="13" font-weight="700" fill="#1c2430"','l':'fill="none" stroke="#3d4654" stroke-width="1.5" marker-end="url(#a)"'}
s=re.sub(r'<style>.*?</style>','',s,flags=re.S)
s=s.replace('class="n" x="30" y="234" fill="#2f6db3"','font-size="12" fill="#2f6db3" x="30" y="234"')
s=re.sub(r'class="(\w+)"',lambda m:A[m.group(1)],s)
open(p,'w',encoding='utf8').write(s)
