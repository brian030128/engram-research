import resource,subprocess,sys,time
t=time.time(); r=subprocess.run(sys.argv[1:]); u=resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"exit={r.returncode} wall_s={time.time()-t:.1f} user_s={u.ru_utime:.1f} sys_s={u.ru_stime:.1f} maxrss_gb={u.ru_maxrss/1048576:.2f}",file=sys.stderr)
