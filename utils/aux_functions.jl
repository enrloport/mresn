import XLSX


function swapcols!(X::AbstractMatrix, i::Integer, j::Integer)
    l = size(X,1)
    @inbounds for k = 1:l
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
end

function digits_pi()
    open("pidigits.txt") do f
        nf = open("digits.txt", "a")

        # while ! eof(f)
        line = 0
        while line < 100
            line += 1
            s = readline(f)
            for c in s
                write(nf, string(c,"\n"))
            end
        end
        close(nf)    
    end
end


function xlsx_to_txt(input_file, output_file)
    xf          = XLSX.readxlsx(input_file)
    sheet_names = XLSX.sheetnames(xf)
    sheet       = xf[sheet_names[1]]
    data        = sheet["B13:M87"]

    new_file = open(output_file, "a")
    for d in data
        if isa(d, AbstractFloat)
            write(new_file, string(d,"\n"))
        end
    end
    close(new_file)
end

xlsx_to_txt("unemplyment.xlsx", "unemplyment.txt")



