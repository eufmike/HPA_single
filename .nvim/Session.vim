let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/send_qsub.py
badd +25 /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/resnet_multicls_multigpu.py
argglobal
%argdel
edit /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/resnet_multicls_multigpu.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 30 + 105) / 210)
exe '2resize ' . ((&lines * 31 + 24) / 48)
exe 'vert 2resize ' . ((&columns * 179 + 105) / 210)
exe '3resize ' . ((&lines * 13 + 24) / 48)
exe 'vert 3resize ' . ((&columns * 179 + 105) / 210)
argglobal
enew
file NvimTree_1
balt /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/send_qsub.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
argglobal
balt /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/send_qsub.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 26 - ((25 * winheight(0) + 15) / 31)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 26
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("term:///dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single//238106:/bin/bash", ":p")) | buffer term:///dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single//238106:/bin/bash | else | edit term:///dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single//238106:/bin/bash | endif
if &buftype ==# 'terminal'
  silent file term:///dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single//238106:/bin/bash
endif
balt /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/resnet_multicls_multigpu.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 47 - ((11 * winheight(0) + 6) / 13)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 47
normal! 071|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 30 + 105) / 210)
exe '2resize ' . ((&lines * 31 + 24) / 48)
exe 'vert 2resize ' . ((&columns * 179 + 105) / 210)
exe '3resize ' . ((&lines * 13 + 24) / 48)
exe 'vert 3resize ' . ((&columns * 179 + 105) / 210)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
