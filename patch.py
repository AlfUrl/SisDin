import sys

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if '# Panel de grupos de riesgo: las tarjetas SON los botones' in line:
        start_idx = i - 1
    if '# Se eliminó la sección "¿Qué hacer ahora?"' in line:
        end_idx = i - 1

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx]
    
    new_lines.append('# ----------------------------------------------------------------\n')
    new_lines.append('# Panel de grupos de riesgo: las tarjetas SON los botones\n')
    new_lines.append('# Los grupos MULTIPLICAN el ICA general (multicontaminante)\n')
    new_lines.append('# ----------------------------------------------------------------\n')
    new_lines.append('@st.fragment\n')
    new_lines.append('def render_risk_groups(ica_general_base):\n')
    
    for i in range(start_idx + 4, end_idx):
        if lines[i].strip() == '':
            new_lines.append(lines[i])
        else:
            new_lines.append('    ' + lines[i])
            
    new_lines.append('\nrender_risk_groups(ica_general_base)\n\n')
    new_lines.extend(lines[end_idx:])
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print('Patched app.py successfully')
else:
    print('Could not find start or end index')
    print('start_idx:', start_idx, 'end_idx:', end_idx)
