Import("env")

model_dir = env.subst("$PROJECT_DIR") + "/model"
cpppath = env.get("CPPPATH", [])
if model_dir not in cpppath:
    cpppath.append(model_dir)
    env.Replace(CPPPATH=cpppath)
print("[PIO] Added model include path:", model_dir)